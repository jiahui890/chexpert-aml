import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Activation, Add, \
    AveragePooling2D, BatchNormalization, Conv2D, \
    Dense, Dropout, Flatten, GlobalMaxPooling2D, Input, \
    Lambda, MaxPooling2D, MaxPool2D, ZeroPadding2D
from keras.initializers import glorot_uniform

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
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    #element wise addition directly from the input (hence identity block)
    X = Add()([X, X_shortcut])  # SKIP Connection
    X = Activation('relu')(X)

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
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    # shortcut branch is a single conv (hence convolution block)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet152_new(output_size, not_transfer=False, feature_shape=(4,), image_shape=(320,320, 1)):

    inputs_feature = Input(shape=feature_shape)
    inputs_image = Input(shape=image_shape)

    # define resnet layers
    x = ZeroPadding2D((3, 3))(inputs_image)
    
    #first conv layer
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    #conv2_x (3 blocks)
    x = convolutional_block(x, 3, [64, 64, 256], stage=2, block='a', s=2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    #conv3_x (8 blocks)
    x = convolutional_block(x, 3, [128, 128, 512], stage=3, block='a', s=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='e')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='f')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='g')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='h')
    
    #conv4_x (36 blocks)
    x = convolutional_block(x, 3, [256, 256, 1024], stage=4, block='a', s=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='g')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='h')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='i')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='j')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='k')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='l')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='m')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='n')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='o')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='p')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='q')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='r')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='s')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='t')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='u')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='v')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='x')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='y')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='z')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='aa')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='ab')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='ac')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='ad')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='ae')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='af')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='ag')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='ah')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='ai')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='aj')
    
    #conv5_x (3 blocks)
    x = convolutional_block(x, 3, [512, 512, 2048], stage=5, block='a', s=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    # create classification layers, first flatten the convolution output
    x = Flatten()(x)

    # create hidden layers for classification
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # create output layer
    x = Dense(output_size)(x)
    x = Activation("sigmoid", name='predicted_observations')(
        x)  # sigmoid and not softmax because we are doing multi-label

    model = Model(inputs=[inputs_feature, inputs_image],
                  outputs=x,
                  name='ResNet152_new')

    return model

#keras standard MobileNet v2
def MobileNetv2_keras(output_size, not_transfer=False, feature_shape=(4,), image_shape=(320,320, 3)):

    cnn_base = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=not_transfer,
                                                              weights='imagenet')
    cnn_base.trainable = not_transfer

    inputs_feature = tf.keras.Input(shape=feature_shape)
    inputs_image = tf.keras.Input(shape=image_shape)
    x = cnn_base(inputs_image, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Concatenate()([x, inputs_feature])
    x = tf.keras.layers.Activation('relu')(x)

    # create output layer
    x = tf.keras.layers.Dense(output_size)(x)
    x = tf.keras.layers.Activation("sigmoid", name='predicted_observations')(
        x)

    # create model class
    model = tf.keras.Model(inputs=[inputs_feature, inputs_image], 
                           outputs=x,
                           'MobileNetv2_keras')

    return model

# keras standard DenseNet121
def DenseNet121_keras(output_size, not_transfer=False, feature_shape=(4,), image_shape=(320,320, 3)):

    cnn_base = tf.keras.applications.DenseNet121(include_top=not_transfer,
                                                 weights='imagenet')
    cnn_base.trainable = not_transfer

    #create 2 input layers, one for img and one for non-img
    inputs_feature = tf.keras.Input(shape=feature_shape)
    inputs_image = tf.keras.Input(shape=image_shape)
    
    #use densenet for the img
    x1 = cnn_base(inputs_image, training=False)
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Flatten()(x1)
    
    #branch 2 for the non-image features
    x2 = Dense()(inputs_feature)
    x2 = Activation("relu")(x2)
    x2 = Dropout(0.5)(x2)
    
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Activation('relu')(x)

    # create output layer
    x = tf.keras.layers.Dense(output_size)(x)
    x = tf.keras.layers.Activation("sigmoid", name='predicted_observations')(
        x)

    # create model class
    model = tf.keras.Model(inputs=[inputs_feature, inputs_image], 
                           outputs=x,
                           name = 'DenseNet121_keras')

    return model

# keras standard ResNet152
def ResNet152_keras(output_size, not_transfer=False, feature_shape=(4,), image_shape=(320,320, 3)):

    cnn_base = tf.keras.applications.ResNet152(include_top=not_transfer,
                                                 weights='imagenet')
    cnn_base.trainable = not_transfer

    #create 2 input layers, one for img and one for non-img
    inputs_feature = tf.keras.Input(shape=feature_shape)
    inputs_image = tf.keras.Input(shape=image_shape)
    
    #use densenet for the img
    x1 = cnn_base(inputs_image, training=False)
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Flatten()(x1)
    
    #branch 2 for the non-image features
    x2 = Dense()(inputs_feature)
    x2 = Activation("relu")(x2)
    x2 = Dropout(0.5)(x2)
    
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Activation('relu')(x)

    # create output layer
    x = tf.keras.layers.Dense(output_size)(x)
    x = tf.keras.layers.Activation("sigmoid", name='predicted_observations')(
        x)

    # create model class
    model = tf.keras.Model(inputs=[inputs_feature, inputs_image], 
                           outputs=x,
                           name='ResNet152_keras')

    return model


cnn_models = {
    "ResNet152_new": ResNet152_new,
    "MobileNetv2_keras": MobileNetv2_keras,
    "Densenet121_keras": DenseNet121,
    "ResNet152_keras": ResNet152_new
}
