import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Activation, Add, \
    AveragePooling2D, BatchNormalization, Conv2D, \
    Dense, Dropout, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Input, Lambda, MaxPooling2D, MaxPool2D, ZeroPadding2D, Concatenate
from keras.initializers import glorot_uniform

# <----- RESNET UTILITY FUNCTIONS ----->
# define function to create identity blocks
def identity_block(X, f, filters, stage, block):
    #for referencing blocks
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #extract the number of filters from array input
    F1, F2, F3 = filters
    
    #first make a copy of the input (used to skip)
    X_shortcut = X
    
    #run the convolutions
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    #element wise addition directly from the input (hence identity block)
    X = tf.keras.layers.Add()([X, X_shortcut])  # SKIP Connection
    X = tf.keras.layers.Activation('relu')(X)

    return X


# define function to create convolutional blocks (not used)
def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #extract the number of filters from array input
    F1, F2, F3 = filters
    
    #first make a copy of the input (used to skip)
    X_shortcut = X
    
    #run the convolutions
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
                               kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                               kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                               kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    # shortcut branch is a single conv (hence convolution block)
    X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X

def ResNet152_new(output_size, 
                  not_transfer=False, 
                  feature_shape=(4,), 
                  image_shape=(320,320, 1)):

    inputs_feature = tf.keras.layers.Input(shape=feature_shape)
    inputs_image = tf.keras.layers.Input(shape=image_shape)
    
    # define branch 1
    # define resnet layers
    x1 = tf.keras.layers.ZeroPadding2D((3, 3))(inputs_image)
    
    #first conv layer
    x1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(x1)
    x1 = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x1)

    #conv2_x (3 blocks)
    x1 = convolutional_block(x1, 3, [64, 64, 256], stage=2, block='a', s=2)
    x1 = identity_block(x1, 3, [64, 64, 256], stage=2, block='b')
    x1 = identity_block(x1, 3, [64, 64, 256], stage=2, block='c')
    
    #conv3_x (8 blocks)
    x1 = convolutional_block(x1, 3, [128, 128, 512], stage=3, block='a', s=2)
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='b')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='c')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='d')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='e')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='f')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='g')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='h')
    
    #conv4_x (36 blocks)
    x1 = convolutional_block(x1, 3, [256, 256, 1024], stage=4, block='a', s=2)
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='b')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='c')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='d')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='e')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='f')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='g')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='h')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='i')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='j')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='k')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='l')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='m')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='n')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='o')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='p')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='q')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='r')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='s')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='t')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='u')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='v')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='x')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='y')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='z')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='aa')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='ab')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='ac')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='ad')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='ae')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='af')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='ag')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='ah')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='ai')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='aj')
    
    #conv5_x (3 blocks)
    x1 = convolutional_block(x1, 3, [512, 512, 2048], stage=5, block='a', s=2)
    x1 = identity_block(x1, 3, [512, 512, 2048], stage=5, block='b')
    x1 = identity_block(x1, 3, [512, 512, 2048], stage=5, block='c')

    x1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x1)

    # create classification layers, first flatten the convolution output
    x1 = tf.keras.layers.Flatten()(x1)

    #build second branch for non-image features
    x2 = tf.keras.layers.Dense(10)(inputs_feature)
    x2 = tf.keras.layers.Activation("relu")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    
    #conatenate the features
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Activation('relu')(x)
    
    # create hidden layers for classification
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # create output layer
    x = tf.keras.layers.Dense(output_size)(x)
    x = tf.keras.layers.Activation("sigmoid", name='predicted_observations')(x)  # sigmoid and not softmax because we are doing multi-label

    
    model = tf.keras.Model(inputs=[inputs_feature, inputs_image],
                                  outputs=x,
                                  name='ResNet152_new')

    return model

def DenseNet121_new(output_size, 
                    not_transfer=False, 
                    feature_shape=(3,), 
                    image_shape=(320,320, 1),
                    filters = 32):

    inputs_feature = tf.keras.layers.Input(shape=feature_shape)
    inputs_image = tf.keras.layers.Input(shape=image_shape)
    
    #utility functions for densenet
    #batch norm + relu + conv
    def bn_rl_conv(x,filters,kernel=1,strides=1):

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, kernel, strides=strides,padding = 'same')(x)
        return x
    
    def dense_block(x, repetition):
        
        for _ in range(repetition):
            y = bn_rl_conv(x, 4*filters)
            y = bn_rl_conv(y, filters, 3)
            x = tf.keras.layers.Concatenate()([y,x])
        return x
        
    def transition_layer(x):
        
        x = bn_rl_conv(x, keras.backend.int_shape(x)[-1] //2 )
        x = tf.keras.layers.AveragePooling2D(2, strides = 2, padding = 'same')(x)
        return x

    #build branch 1 for img data
    x1 = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = 'same')(inputs_image)
    x1 = tf.keras.layers.MaxPool2D(3, strides = 2, padding = 'same')(x1)
    
    for repetition in [6,12,24,16]:   
        d = dense_block(x1, repetition)
        x1 = transition_layer(d)
    
    x1 = tf.keras.layers.GlobalAveragePooling2D()(d)
    x1 = tf.keras.layers.Flatten()(x1)
    
    #branch 2 for the non-image features
    x2 = tf.keras.layers.Dense(10)(inputs_feature)
    x2 = tf.keras.layers.Activation("relu")(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    
    #conatenate the features
    x = tf.keras.layers.Concatenate()([x1, x2])    
    x = tf.keras.layers.Activation('relu')(x)
    
    # create hidden layers for classification
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # create output layer
    x = tf.keras.layers.Dense(output_size)(x)
    x = tf.keras.layers.Activation("sigmoid", name='predicted_observations')(x)

    # create model class
    model = tf.keras.Model(inputs=[inputs_feature, inputs_image], 
                           outputs=x,
                           name = 'DenseNet121_keras')

    return model

#keras standard MobileNet v2
def MobileNetv2_keras(output_size, 
                      not_transfer=False, 
                      feature_shape=(4,), 
                      image_shape=(320,320, 3)):

    cnn_base = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=not_transfer,
                                                              weights='imagenet')
    cnn_base.trainable = not_transfer
    
    #inputs
    inputs_feature = tf.keras.Input(shape=feature_shape)
    inputs_image = tf.keras.Input(shape=image_shape)
    
    x1 = cnn_base(inputs_image, training=False)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    
    #branch 2 for the non-image features
    x2 = tf.keras.layers.Dense(10)(inputs_feature)
    x2 = tf.keras.layers.Activation("relu")(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    
    #conatenate the features
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Activation('relu')(x)

    # create output layer
    x = tf.keras.layers.Dense(output_size)(x)
    x = tf.keras.layers.Activation("sigmoid", name='predicted_observations')(x)

    # create model class
    model = tf.keras.Model(inputs=[inputs_feature, inputs_image], 
                           outputs=x,
                           name = 'MobileNetv2_keras')

    return model

# keras standard DenseNet121
def DenseNet121_keras(output_size, 
                      not_transfer=False, 
                      feature_shape=(4,), 
                      image_shape=(320,320, 3)):

    cnn_base = tf.keras.applications.DenseNet121(include_top=not_transfer,
                                                 weights='imagenet')
    cnn_base.trainable = not_transfer

    #create 2 input layers, one for img and one for non-img
    inputs_feature = tf.keras.Input(shape=feature_shape)
    inputs_image = tf.keras.Input(shape=image_shape)
    
    #use densenet for the img
    x1 = cnn_base(inputs_image, training=False)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    
    #branch 2 for the non-image features
    x2 = tf.keras.layers.Dense(10)(inputs_feature)
    x2 = tf.keras.layers.Activation("relu")(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    
    #conatenate the features
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Activation('relu')(x)

    # create output layer
    x = tf.keras.layers.Dense(output_size)(x)
    x = tf.keras.layers.Activation("sigmoid", name='predicted_observations')(x)

    # create model class
    model = tf.keras.Model(inputs=[inputs_feature, inputs_image], 
                           outputs=x,
                           name = 'DenseNet121_keras')

    return model

# keras standard ResNet152
def ResNet152_keras(output_size, 
                    not_transfer=False, 
                    feature_shape=(4,), 
                    image_shape=(320,320, 3)):

    cnn_base = tf.keras.applications.ResNet152(include_top=not_transfer,
                                                 weights='imagenet')
    cnn_base.trainable = not_transfer

    #create 2 input layers, one for img and one for non-img
    inputs_feature = tf.keras.Input(shape=feature_shape)
    inputs_image = tf.keras.Input(shape=image_shape)
    
    #use densenet for the img
    x1 = cnn_base(inputs_image, training=False)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    
    #branch 2 for the non-image features
    x2 = tf.keras.layers.Dense(10)(inputs_feature)
    x2 = tf.keras.layers.Activation("relu")(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    
    #concatenate the branches
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Activation('relu')(x)

    # create output layer
    x = tf.keras.layers.Dense(output_size)(x)
    x = tf.keras.layers.Activation("sigmoid", name='predicted_observations')(x)

    # create model class
    model = tf.keras.Model(inputs=[inputs_feature, inputs_image], 
                           outputs=x,
                           name='ResNet152_keras')

    return model


cnn_models = {
    "ResNet152_new": ResNet152_new,
    "DenseNet121_new": DenseNet121_new,
    "MobileNetv2_keras": MobileNetv2_keras,
    "DenseNet121_keras": DenseNet121_keras,
    "ResNet152_keras": ResNet152_keras
}
