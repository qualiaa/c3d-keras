# C3D Keras

An implementation of [C3D](http://github.com/facebook/C3D) for Keras.

## Weights

The [original C3D weight file](https://www.dropbox.com/s/vr8ckp0pxgbldhs/conv3d_deepnetA_sport1m_iter_1900000?dl=0) can be converted by calling `caffe2keras.py`, or you can use the existing set converted by [albertomontesg](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2) available [here](https://www.dropbox.com/s/ypiwalgtlrtnw8b/c3d-sports1M_weights.h5?dl=0)

## Data

To evaluate on one video, use [any
video](https://www.youtube.com/watch?v=dM06AMFLsrc) from the [sports1m dataset](https://github.com/gtoderici/sports-1m-dataset/). To acquire more of the dataset, use [sports1m-get](https://github.com/qualiaa/sports1m-get)
