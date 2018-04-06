from __future__ import print_function

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, manual_variable_initialization
# TODO : check gpu configuration
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import os.path

# from EmotiW2018.ImageBased.DenseNet import densenet
import densenet
import numpy as np

from keras.layers import Dense, Flatten, Dropout, Conv2D
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import model_from_json

#TODO : check batch size
batch_size = 20
nb_classes = 7
nb_epoch = 100
nb_train_samples = 32298
nb_test_samples = 3589

img_rows, img_cols = 224, 224
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation
hidden_layer = 4096

base_model = densenet.DenseNetImageNet169(img_dim, classes=nb_classes, include_top=False, weights='imagenet')
x = base_model.layers[-2].output
x = Flatten()(x)
x = Dense(hidden_layer, activation='relu', name='fc1')(x)
#x = Dense(hidden_layer, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
out = Dense(nb_classes, activation='softmax', name='predicton')(x)
model = Model(inputs=base_model.input, outputs=out)

#x = base_model.layers[-2].output
#x = Conv2D(512, (1, 1), activation='relu', padding='same', use_bias=False, name='dim_reduction')(x)
#x = Flatten(name='flatten')(x)
#x = Dense(hidden_layer, activation='relu', name='fc1')(x)
#x = Dropout(0.5)(x)
##x = Dense(hidden_layer, activation='relu', name='fc2')(x)
#out = Dense(nb_classes, activation='softmax', name='prediction')(x)
#model = Model(inputs=base_model.input, outputs=out)

# TODO : check specific network model
# base_model = densenet.DenseNetImageNet161(img_dim, classes=nb_classes, include_top=False)
# x = base_model.output
# out = Dense(nb_classes, activation='softmax', name='prediction')(x)
# model = Model(inputs=base_model.input, outputs=out)
#
# base_model = densenet.DenseNetImageNet121(img_dim, classes=nb_classes, include_top=False, weights='imagenet')
#
# # x = base_model.output
# # x = Flatten(name='flatten')(x)
# # x = Dense(hidden_layer, activation='relu', name='fc1')(x)
# # x = Dropout(0.5)(x)
# # x = Dense(hidden_layer, activation='relu', name='fc2')(x)
# # out = Dense(nb_classes, activation='softmax', name='fc3-prediction')(x)
# # model = Model(inputs=base_model.input, outputs=out)

# # up to 2nd dense block model
# x = base_model.get_layer('average_pooling2d_3').output
# x = Flatten(name='flatten')(x)
# x = Dense(hidden_layer, activation='relu', name='fc1')(x)
# x = Dense(hidden_layer, activation='relu', name='fc2')(x)
# x = Dropout(0.5)(x)
# out = Dense(nb_classes, activation='softmax', name='fc3-prediction')(x)
# model = Model(inputs=base_model.input, outputs=out)
# print("Model created")

# temp spyder
#model = model_from_json(open('/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/DenseNet/' + 'DenseNet121_fer_glbavgpool_fc4096' +'/' + 'DenseNet121_fer_glbavgpool_fc4096' + '.json').read())
#model.load_weights('/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/DenseNet/' + 'DenseNet121_fer_glbavgpool_fc4096' + '/' + 'DenseNet121_fer_glbavgpool_fc4096.h5')

model.summary()
optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")


# Load model
# TODO : Check svae file name
TRAIN_DIR = '/home/dmsl/nas/DMSL/FER2013/Images/Training'
TEST_DIR = '/home/dmsl/nas/DMSL/FER2013/Images/PrivateTest'
dir_name = 'DenseNet121_fc2_drop'
save_path = '/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/DenseNet/' + dir_name
weights_file = save_path + "/" + dir_name + ".h5"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(save_path + '/graph'):
    os.mkdir(save_path + '/graph')

# model.load_weights(weights_file)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    class_mode='categorical'
)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                    cooldown=0, patience=10, min_lr=1e-8, verbose=1)
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                  save_weights_only=True, verbose=1)

tb_hist = TensorBoard(
        log_dir=save_path + '/graph',
        histogram_freq=0,
        write_graph=True,
        write_images=True
        )

callbacks = [lr_reducer, model_checkpoint, tb_hist]

# save model
open(save_path + '/' + dir_name + '.json', 'w').write(model.to_json())
print("Start training!")

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=nb_epoch,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size,
    callbacks=callbacks,
    verbose=1
)