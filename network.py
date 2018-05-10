#!/usr/bin/env python3

DATA_DIR="data"
MY_WEIGHTS=True

import sys
from pathlib import Path, PurePath
from itertools import permutations, product, repeat, takewhile

import cv2
import pims
import numpy as np

import pkl_xz
from random_crop_slice import *
import my_weights as mine, his_weights as his

import keras
import keras.utils.vis_utils as vis
from keras import backend as K
from keras.layers import Conv2D, Conv3D, Flatten, Input, Reshape, Lambda, concatenate
from keras.layers import Dense, Dropout,ZeroPadding3D
from keras.layers import MaxPooling3D, UpSampling2D,UpSampling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.metrics import top_k_categorical_accuracy as top_k
from keras.models import Model

TF = [True,False]

from functools import reduce
apply = lambda f, x: f(x)
flip = lambda f: lambda a, b: f(b, a)
apply_sequence = lambda l, x: reduce(flip(apply), l, x)

def generate_settings():
    # generate all possible weight configuration settings
    #transpositions=([xyz + (3,4) for xyz in permutations((0,1,2))] if MY_WEIGHTS
              #else [xyz + (1,0) for xyz in permutations((2,3,4))])
    transposition=[(0,1,2,3,4) if MY_WEIGHTS else (2,3,4,1,0)]
    all_results = []
    #keys = ["transposition","flip_x","flip_y","flip_z","flip_conv_bias","flip_fc_bias","flip_fc"]
    keys = ["transposition",
            "flip_x",
            "flip_y",
            "flip_z",
            "flip_conv_bias",
            "flip_fc_bias"]
    values_list = product(transposition,*repeat(TF,len(keys)-1))
    search_space = [dict(zip(keys,a)) for a in values_list]
    return search_space

def _output_suf():
    return (("my_" if MY_WEIGHTS else "his_") + "weights.pkl.xz")


def create_model():
    data_format = "channels_first"

    C3 = lambda filter_size: Conv3D(
            filter_size,
            (3,3,3),
            data_format=data_format,
            activation="relu",
            padding="same")
    def P3(shape=(2,2,2)):
        return MaxPooling3D(
            shape,
            data_format=data_format)

    coarse_architecture = [
        # encoder
        C3(64), P3((1,2,2)),
        C3(128), P3(),
        C3(256), C3(256), P3(),
        C3(512), C3(512), P3(),
        C3(512), C3(512),
        ZeroPadding3D(padding=(0,1,1),data_format=data_format),
        P3(),
        Flatten(),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(487,activation="softmax")
    ]

    input_shape = (16,112,112)
    input_shape = (3,) + input_shape

    cropped_input = Input(shape=input_shape,dtype='float32',name="cropped_input")
    cropped_output = apply_sequence(coarse_architecture,cropped_input)

    # Build model
    def top_3(ytrue,ypred): return top_k(ytrue,ypred,3)
    def top_5(ytrue,ypred): return top_k(ytrue,ypred,5)

    model = Model(inputs=[cropped_input],
                  outputs=[cropped_output])
    print("Compiling model")
    model.compile(loss="categorical_crossentropy",
            optimizer="sgd",
            metrics=['categorical_accuracy',top_3,top_5])
    return model

def load_weights(settings):
    if MY_WEIGHTS:
        params = mine.load_weights(settings)
    else:
        params = his.load_weights(settings)
    for l,p in params.items():
        model.layers[l].set_weights(p)

def pims_to_np(pims_clip):
    clip = np.stack(pims_clip[:],axis=0)
    try:
        crop_slice = random_crop_slice(clip.shape,(16,112,112,3))
    except AssertionError as e:
        print(e.args)
        print("For shapes:")
        print(len(pims_clip))
        print(clip.shape)
        raise e

    clip = clip[crop_slice]
    clip = clip.transpose(3,0,1,2)
    clip = np.array(clip,dtype=np.float32)
    #clip /= 255.0

    return clip

# take average prediction over five clips for improved classification
def evaluate(num_vids=None):
    root = PurePath(DATA_DIR,"train","videos")
    if num_vids is None:
        with open(DATA_DIR+"/num_vids") as f:
            num_vids=int(f.read())

    X_train = []
    Y_train = []
    Y_label = []
    file_ids = []
    clips_per_vid = 5

    print("Reading {:d} video files".format(num_vids))
    with open(DATA_DIR+"/labels.txt") as label_file:
        i = 0
        num_read = 0
        while num_read < num_vids:
            label_strings = label_file.readline().split(',')
            label = int(label_strings[0]) # ignore multiclass for now
    
            video_path = Path(root,"{:06d}".format(i))
            i+=1
            if not video_path.is_dir(): continue
            clips = [pims.open(str(clip)+"/*png") for clip in video_path.iterdir()]
            if len(clips) != clips_per_vid: continue
            if any(len(clip) < 16 for clip in clips): continue
    
            X_train.extend(clips)
            Y_label.extend([label]*len(clips))
            file_ids.extend([i-1]*len(clips))
            num_read+=1
            if num_read % 1000 == 0: print(num_read,"/",num_vids)
    Y_train = keras.utils.np_utils.to_categorical(Y_label,487)
    Y_label = np.array(Y_label)
    
    
    batch_size = 50
    n_batches = len(X_train) // batch_size
    
    def examples_generator(batch_size,steps):
        processed = 0
        gen=zip(map(pims_to_np,X_train),Y_train)
        for i in range(steps):
            batch=[]
            for j in range(batch_size):
                try:
                    batch.append(next(gen))
                except AssertionError:
                    print("For file: {:06d}".format(file_ids[processed]))
                    sys.exit(1)
                processed += 1

            batch = tuple(map(np.stack,zip(*batch)))
            yield batch

    print("Evaluating results")
    try: 
        pred = model.predict_generator(
                examples_generator(batch_size,n_batches),
                steps=n_batches)
                #workers=6,
                #use_multiprocessing=True)
    except Exception as e:
        print(type(e))
        print(str(e.args)[:200])
        sys.exit(1)

    print(len(pred))

    Y_label=Y_label[:n_batches*batch_size]
    avg_pred = np.vstack(x.mean(axis=0) for x in
            np.vsplit(pred,len(pred)//5))
    avg_label = np.array(Y_label[::5])

    def get_scores(preds,label):
        #n = preds.shape[0]
        ranked_preds = np.argsort(np.argsort(preds,axis=1),axis=1)
        label_ranks = ranked_preds[np.arange(len(label)),label]
        label_ranks = (preds.shape[1] - 1) - label_ranks
        return tuple(np.less(label_ranks,k).mean() for k in (1,3,5))
    
    scores = get_scores(pred,Y_label)
    avg_scores = get_scores(avg_pred,avg_label)

    return scores, avg_scores


def test_one_video(model,path):
    print("Loading example video")
    cap = cv2.VideoCapture(str(path))
    vid = []
    while True:
        ret, img = cap.read()
        if not ret: break
        vid.append(cv2.resize(img,(171,128)))
    vid = np.array(vid,dtype=np.float32)

    X = vid[2000:2016,8:8+112,30:30+112,:]
    X = np.moveaxis(X,3,0)
    del vid
    X = np.expand_dims(X,0)

    output = model.predict_on_batch(X)
    output = np.mean(output,axis=0)

    results = sorted(enumerate(output),key=lambda a:-a[1])

    print(results[:3])

    with open("labels.txt") as f:
        labels = [l[:-1] for l in list(f)]

    for i,_ in results[:3]:
        print(i,labels[i])
    return results

def quick_eval_all_settings():
    results = []
    search_space = generate_settings()
    for i,settings in enumerate(search_space):
        print("Trying setting {:d} out of {:d}".format(i,len(search_space)))
        print(settings)
        load_weights(settings)
        scores = evaluate(50)
        results.append((i,settings,scores))
        print(scores)

    pkl_xz.save(results,"quick_eval_" + _output_suf())
    return results

def find_good_settings():
    results = quick_eval_all_settings()

    good_settings = []

    # sort resuts by the top 5 result for averaged predictions, descending
    sorted_results = sorted(results,key=lambda a: -a[2][1][2])
    good_settings = list(takewhile(lambda a: a[2][1][2] >= 0.6,sorted_results))

    pkl_xz.save(good_settings,"good_settings_" + _output_suf())
    print("Good settings: {}".format(good_settings))

def eval_good_settings():
    good_settings = pkl_xz.load("good_settings_" + _output_suf())

    results = []
    for _,settings,_ in good_settings:
        load_weights(settings)
        scores = evaluate(500)
        results.append((settings,scores))
        print(scores)

    pkl_xz.save(results,"eval_good_settings" + _output_suf())
    return results
    
model = create_model()
load_weights()
scores = evaluate()
print(scores)
