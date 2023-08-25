import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)
def get_model(hyper_params):
    """Generating rpn model for given hyper params. one channel
    inputs:
        hyper_params = dictionary
    outputs:
        rpn_model = tf.keras.model
        feature_extractor = feature extractor layer from the base model
    """
    img_size = hyper_params["img_size_vgg"]
    img_input = layers.Input(shape=img_size)

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    # print("WOOOW")
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(x)
    # print("WOOOW1")
    rpn_cls_output = Conv2D(hyper_params["anchor_count"], (1, 1), activation="sigmoid", name="rpn_cls")(x)
    # print("WOOOW2")
    rpn_reg_output = Conv2D(hyper_params["anchor_count"] * 4, (1, 1), activation="linear", name="rpn_reg")(x)
    # print("WOOOW3")
    rpn_model = Model(inputs=img_input, outputs=[rpn_reg_output, rpn_cls_output])
    feature_extractor = rpn_model.get_layer("block5_conv3")
    return rpn_model, feature_extractor

def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model
    """
    model(tf.random.uniform((1, 896,896, 1)))
