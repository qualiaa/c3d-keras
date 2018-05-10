import h5py
import numpy as np

def load_weights(settings=None):
    if settings is None:
        settings = good_settings[0]
    print("Loading model parameters")
    model_layers = [1,3,5,6,8,9,11,12,16,18,20]
    conv_input   = [0,2,4,5,7,8,10,11]
    fc_input     = [15,17,19]
    input_layers = conv_input + fc_input

    params_file = h5py.File("c3d-sports1M_weights.h5")
    output_params = {}

    for i,l in zip(input_layers,model_layers):
        layer = params_file["layer_{:d}".format(i)]
        w = np.array(layer["param_0"])
        b = np.array(layer["param_1"])
        if i in conv_input:
            if settings["flip_z"]:
                w = np.flip(w,2)
            if settings["flip_y"]:
                w = np.flip(w,3)
            if settings["flip_x"]:
                w = np.flip(w,4)
            if settings["flip_conv_bias"]:
                b = np.flipud(b)
            w = w.transpose(settings["transposition"])
        else:
            if settings["flip_fc_bias"]:
                b = np.flipud(b)
        output_params[l]=(w,b)

    params_file.close()
    del params_file
    return output_params

good_settings=[
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': False, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': True, 'flip_fc_bias': False},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': True, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': False, 'flip_fc_bias': False},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': False, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': False, 'flip_fc_bias': False},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': True, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': True, 'flip_fc_bias': True},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': True, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': True, 'flip_fc_bias': False},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': True, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': False, 'flip_fc_bias': False},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': True, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': False, 'flip_fc_bias': True},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': False, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': False, 'flip_fc_bias': False},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': False, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': False, 'flip_fc_bias': True},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': False, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': True, 'flip_fc_bias': False},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': False, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': True, 'flip_fc_bias': True},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': True, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': False, 'flip_fc_bias': True},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': True, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': True, 'flip_fc_bias': False},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': True, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': True, 'flip_fc_bias': True},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': False, 'flip_y': True, 'flip_z': True,
         'flip_conv_bias': False, 'flip_fc_bias': True},
        {'transposition': (2, 3, 4, 1, 0),
         'flip_x': False, 'flip_y': True, 'flip_z': False,
         'flip_conv_bias': True, 'flip_fc_bias': True}
]
