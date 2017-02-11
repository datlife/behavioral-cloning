from keras.models import Model
from keras.layers import Input, Lambda, Dropout, merge
from keras.layers import Activation, BatchNormalization, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, Cropping2D


def rnpa_bottleneck_layer(input_tensor, nb_filters, filter_sz, stage,
                          init='glorot_normal', reg=0.0, use_shortcuts=True):
    '''
    Pre-activation Network

    Perform batch (norm ---> Activation -- > Conv2D)x2 --> merge 'sum'
    '''
    nb_in_filters, nb_bottleneck_filters = nb_filters

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = '+' + str(stage)

    # batch-norm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage > 1:  # first activation is just after conv1
        x = BatchNormalization(axis=1, name=bn_name + 'a')(input_tensor)
        x = Activation('relu', name=relu_name + 'a')(x)
    else:
        x = input_tensor

    x = Convolution2D(
        nb_bottleneck_filters, 1, 1,                # used to 1-1 here
        init=init,
        W_regularizer=l2(reg),
        bias=False,
        name=conv_name + 'a'
    )(x)

    # batch-norm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = BatchNormalization(axis=1, name=bn_name + 'b')(x)
    x = Activation('relu', name=relu_name + 'b')(x)
    x = Convolution2D(
        nb_bottleneck_filters, filter_sz, filter_sz,
        border_mode='same',
        init=init,
        W_regularizer=l2(reg),
        bias=False,
        name=conv_name + 'b'
    )(x)

    # batch-norm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = BatchNormalization(axis=1, name=bn_name + 'c')(x)
    x = Activation('relu', name=relu_name + 'c')(x)
    x = Convolution2D(nb_in_filters, 3, 3, border_mode='same',           # Used to 1-1 here
                      init=init, W_regularizer=l2(reg),
                      name=conv_name + 'c'
                      )(x)
    # merge
    if use_shortcuts:
        x = merge([x, input_tensor], mode='sum', name=merge_name)

    return x


def ResNetPreAct(input_shape=(3, 32, 32), nb_classes=10,
                 layer1_params=(5, 32, 2),
                 res_layer_params=(3, 16, 3),
                 init='glorot_normal', reg=0.0, use_shortcuts=True
                 ):
    """
    Return a new Residual Network using full pre-activation based on the work in
    "Identity Mappings in Deep Residual Networks"  by He et al
    http://arxiv.org/abs/1603.05027
    The following network definition achieves 92.0% accuracy on CIFAR-10 test using
    `adam` optimizer, 100 epochs, learning rate schedule of 1e.-3 / 1.e-4 / 1.e-5 with
    transitions at 50 and 75 epochs:
    ResNetPreAct(layer1_params=(3,128,2),res_layer_params=(3,32,25),reg=reg)

    Removed max pooling and using just stride in first convolutional layer. Motivated by
    "Striving for Simplicity: The All Convolutional Net"  by Springenberg et al
    (https://arxiv.org/abs/1412.6806) and my own experiments where I observed about 0.5%
    improvement by replacing the max pool operations in the VGG-like cifar10_cnn.py example
    in the Keras distribution.

    Parameters
    ----------
    input_dim : tuple of (C, H, W)
    nb_classes: number of scores to produce from final affine layer (input to softmax)
    layer1_params: tuple of (filter size, num filters, stride for conv)
    res_layer_params: tuple of (filter size, num res layer filters, num res stages)
    final_layer_params: None or tuple of (filter size, num filters, stride for conv)
    init: type of weight initialization to use
    reg: L2 weight regularization (or weight decay)
    use_shortcuts: to evaluate difference between residual and non-residual network
    """

    sz_L1_filters, nb_L1_filters, stride_L1 = layer1_params
    sz_res_filters, nb_res_filters, nb_res_stages = res_layer_params

    sz_pool_fin = (input_shape[0] - 30 - 5) / stride_L1

    img_input = Input(shape=input_shape, name='cifar')

    x = Lambda(lambda image: image / 255.0 - 0.5, input_shape=input_shape)(img_input)
    x = Cropping2D(cropping=((30, 5), (1, 1)))(x)
    x = Convolution2D(nb_L1_filters, sz_L1_filters, sz_L1_filters, border_mode='same', subsample=(stride_L1, stride_L1),
                      init=init, W_regularizer=l2(reg), bias=False, name='conv0')(x)
    x = BatchNormalization(axis=1, name='bn0')(x)
    x = Activation('relu', name='relu0')(x)

    # Bottle Neck Layers
    for stage in range(1, nb_res_stages + 1):
        x = rnpa_bottleneck_layer(x, (nb_L1_filters, nb_res_filters), sz_res_filters, stage,
                                  init=init, reg=reg, use_shortcuts=use_shortcuts)
    x = BatchNormalization(axis=1, name='bnF')(x)
    x = Activation('relu', name='reluF')(x)
    x = AveragePooling2D((sz_pool_fin, sz_pool_fin), name='avg_pool')(x)
    x = Flatten(name='flat')(x)
    x = Dense(1024, name='fc1', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, name='fc2', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='output')(x)

    return Model(input=img_input, output=x)
