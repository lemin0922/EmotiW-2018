from __future__ import print_function

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, manual_variable_initialization
# TODO : selecet gpu device
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "1"
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import os.path

from EmotiW2018.ImageBased.DenseNet import densenet
from EmotiW2018.ImageBased.DenseNet.util import Util as util
# from EmotiW2018.temp.MultipleLearningRate import LR_SGD
import numpy as np
import sklearn.metrics as metrics

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.engine import Model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import model_from_json

# TODo : choose batch size
batch_size = 128
nb_classes = 7
nb_epoch = 100

img_rows, img_cols = 224, 224
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 100
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation
hidden_layer = 4096

# TODO : select densenet size
base_model = densenet.DenseNetImageNet121(img_dim, classes=nb_classes, include_top=False, weights=None)
x = base_model.layers[-2].output
x = Flatten(name='flatten')(x)
x = Dense(hidden_layer, activation='relu', name='fc1')(x)
# x = Dropout(0.5)(x)
x = Dense(hidden_layer, activation='relu', name='fc2')(x)
# x = Dropout(0.5)(x)
out = Dense(nb_classes, activation='softmax', name='fc3-prediction')(x)
model = Model(inputs=base_model.input, outputs=out)
print("Model created")

# TODO : choose weigth path
WEIGHT_PATH = '/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/DenseNet/DenseNet121_fer_fc4096x2/DenseNet_DenseNet121_fer_fc4096x2.h5'
model.load_weights(WEIGHT_PATH)
print("Model Load!")
model.summary()

# TODO : new top model
# new_model_input = model.input
# x = model.layers[-3].output
# x = Flatten(name='reshape')(x)
# x = Dense(hidden_layer, activation='relu', name='fc1')(x)
# x = Dropout(0.5, name='dropout1')(x)
# x = Dense(hidden_layer, activation='relu', name='fc2')(x)
# x = Dropout(0.5, name='dropout2')(x)
# out = Dense(nb_classes, activation='softmax', name='new_prediction')(x)
# new_model = Model(inputs=new_model_input, outputs=out)
# new_model.summary()

for layer in model.layers[:-4]:
    layer.trainable = False

# TODO : set learning rate & optimizer
# optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)
optimizer = RMSprop(lr=1e-4, decay=1e-6)
# base_lr = 0.01
# LR_mult_dict = {}
# LR_mult_dict[]
# optimizer_mult = LR_SGD(lr=base_lr, momentum=0.9, decay=0.0, nestrov=False, multiplier=)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

x_train_data = '/home/dmsl/nas/DMSL/AFEW/NpzData/Train_only/x_train_only_mtcnn_color.npz'
y_train_data = '/home/dmsl/nas/DMSL/AFEW/NpzData/Train_only/y_train_only_mtcnn_color.npz'
x_test_data = '/home/dmsl/nas/DMSL/AFEW/NpzData/Val/x_val_mtcnn_color.npz'
y_test_data = '/home/dmsl/nas/DMSL/AFEW/NpzData/Val/y_val_mtcnn_color.npz'

trainX, trainY = util.load_from_npz(x_train_data), util.load_from_npz(y_train_data)
testX, testY = util.load_from_npz(x_test_data), util.load_from_npz(y_test_data)

trainX = trainX.astype('float32')
testX = testX.astype('float32')

# trainX = densenet.preprocess_input(trainX)
# testX = densenet.preprocess_input(testX)

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

# TODO : data augmentation
train_generator = ImageDataGenerator(rescale=1./255,
                                     rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

train_generator.fit(trainX, seed=0)

# Load model
# TODO : select directory name
dir_name = 'DenseNet121_afew_fer_fc4096x2'
save_path = '/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/DenseNet/' + dir_name
if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(save_path + '/graph'):
    os.mkdir(save_path + '/graph')

weights_file = save_path + "/DenseNet_" + dir_name + ".h5"

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                    cooldown=0, patience=5, min_lr=1e-6)
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

model.fit_generator(train_generator.flow(trainX, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(testX, Y_test),
                    validation_steps=testX.shape[0] // batch_size, verbose=2)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

