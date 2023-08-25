"""
Created on Tue Dec 14 14:52:06 2021

@author: HASSINI Houda
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D

def get_model(hyper_params):
    """Generating rpn model for given hyper params using a complex-valued formalism wkth the classical fonctions of tensorflow 
    inputs:
        hyper_params = dictionary
    outputs:
        rpn_model = tf.keras.model
        feature_extractor = feature extractor layer from the base model
    """
    img_size = hyper_params["img_size_vgg"]
    img_input = layers.Input(shape=img_size)

    # Block 1
    x_r = layers.Conv3D(32, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1_r')(img_input)
    x_i = layers.Conv3D(32, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1_i')(img_input)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x_r = layers.Conv3D(32, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2_r')(x)
    x_i = layers.Conv3D(32, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x = layers.MaxPooling3D((1 , 2, 2), strides=(1,2, 2), name='block1_pool')(x)

      # Block 2
    x_r = layers.Conv3D(64, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1_r')(x)
    x_i = layers.Conv3D(64, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x_r = layers.Conv3D(64, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2_r')(x)
    x_i = layers.Conv3D(64, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x = layers.MaxPooling3D((1 , 2, 2), strides=(1,2, 2), name='block2_pool')(x)

# #     # Block 3
    x_r = layers.Conv3D(128, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1_r')(x)
    x_i = layers.Conv3D(128, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x_r = layers.Conv3D(128, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2_r')(x)
    x_i = layers.Conv3D(128, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x_r = layers.Conv3D(128, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3_r')(x)
    x_i = layers.Conv3D(128, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)

    x = layers.MaxPooling3D((1 , 2, 2), strides=(1,2, 2), name='block3_pool')(x)

#     # Block 4
    x_r = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1_r')(x)
    x_i = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x_r = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2_r')(x)
    x_i = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x_r = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3_r')(x)
    x_i = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x = layers.MaxPooling3D((1 , 2, 2), strides=(1,2, 2), name='block4_pool')(x)
#     # Block 5
    x_r = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1_r')(x)
    x_i = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x_r = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2_r')(x)
    x_i = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_r = tf.expand_dims(part_r, axis=1)
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    part_i = tf.expand_dims(part_i, axis=1)
    x = tf.concat([part_r,part_i], axis = 1)
    x_r = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3_r')(x)
    x_i = layers.Conv3D(256, (1,3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3_i')(x)
    part_r = (x_r[:,0,:,:,:]-x_i[:,1,:,:,:])
    part_i = x_r[:,1,:,:,:]+x_i[:,0,:,:,:]
    x = tf.concat([x[:,0,:,:,:],x[:,1,:,:,:]], axis = 3)
    feature_extractor = x
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(x)
    rpn_cls_output = Conv2D(hyper_params["anchor_count"], (1, 1), activation="sigmoid", name="rpn_cls")(x)
    rpn_reg_output = Conv2D(hyper_params["anchor_count"] * 4, (1, 1), activation="linear", name="rpn_reg")(x)
    rpn_model = Model(inputs=img_input, outputs=[rpn_reg_output, rpn_cls_output])
    return rpn_model, feature_extractor


def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model
    """
    model(tf.random.uniform((2, 896,896, 1)))
