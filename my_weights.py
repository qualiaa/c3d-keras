import pkl_xz
import numpy as np

def load_weights(settings=None):
    if settings is None:
        settings = good_settings[0]
    print("Loading model parameters")
    input_weights_string = "keras_weights.pkl.xz"

    conv_layers = [1,3,5,6,8,9,11,12]
    fc_layers = [16,18,20]
    layers = conv_layers + fc_layers
    input_params = pkl_xz.load(input_weights_string+".pkl.xz")
    output_params = {}

    for l,p in zip(layers,input_params):
        w, b = p
        if l in conv_layers:
            if settings["flip_z"]:
                w = np.flip(w,0)
            if settings["flip_y"]:
                w = np.flip(w,1)
            if settings["flip_x"]:
                w = np.flip(w,2)
            if settings["flip_conv_bias"]:
                b = np.flipud(b)
            w = w.transpose(settings["transposition"])
        else:
            if settings["flip_fc_bias"]:
                b = np.flipud(b)
        output_params[l]=(w,b)
    del input_params
    return output_params

good_settings=[
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': False, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': True, 'flip_fc_bias': False},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': False, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': False, 'flip_fc_bias': False},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': True, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': True, 'flip_fc_bias': False},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': True, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': True, 'flip_fc_bias': False},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': True, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': False, 'flip_fc_bias': False},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': True, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': False, 'flip_fc_bias': False},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': True, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': True, 'flip_fc_bias': True},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': False, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': False, 'flip_fc_bias': True},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': False, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': True, 'flip_fc_bias': True},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': False, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': True, 'flip_fc_bias': True},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': False, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': True, 'flip_fc_bias': False},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': False, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': False, 'flip_fc_bias': True},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': True, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': True, 'flip_fc_bias': True},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': False, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': False, 'flip_fc_bias': False},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': True, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': False, 'flip_fc_bias': True},
        {'transposition': (0, 1, 2, 3, 4),
         'flip_x': True, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': False, 'flip_fc_bias': True}
]
