from __future__ import print_function

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, manual_variable_initialization
# TODO : selecet gpu device
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import os.path
from collections import defaultdict
import numpy as np
import cv2

from EmotiW2018.ImageBased.DenseNet import densenet
from EmotiW2018.ImageBased.DenseNet.util import Util as util

from keras.engine import Model
from keras import backend as K
from keras.models import model_from_json

def key2label(key):
    if key == 'angry':
        y_label = 0
    elif key == 'disgust':
        y_label = 1
    elif key == 'fear':
        y_label = 2
    elif key == 'happy':
        y_label = 3
    elif key == 'neutral':
        y_label = 4
    elif key == 'sad':
        y_label = 5
    elif key == 'surprise':
        y_label = 6

    return y_label

def preprocess_img(path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = np.expand_dims(img * 1. / 255, axis=0)

    return img

def check16Frames(frame_list, numframe=16):
    if len(frame_list) == numframe:
        x_frames = np.vstack(frame_list)
        x_frames = np.expand_dims(x_frames, axis=0)
        return x_frames
    else:
        raise ValueError

# TODO : choose model & weigth path
ROOT_PATH = '/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/DenseNet/'
DIR_NAME = 'DenseNet121_fer_fc4096x2'
MODEL_PATH = ROOT_PATH + DIR_NAME + '/' + DIR_NAME +'.json'
WEIGHT_PATH = ROOT_PATH + DIR_NAME + '/' + DIR_NAME + '.h5'

model = model_from_json(open(MODEL_PATH).read())
model.load_weights(WEIGHT_PATH)
print("Model created")

feature = model.get_layer('fc1').output
feature_extract_model = Model(inputs=model.input, outputs=feature)
print("Model Load!")
feature_extract_model.summary()

# TODO : confirm here!
DATA_ROOT = '/home/dmsl/nas/DMSL/AFEW/afew-mtcnn/REARRANGE/Train_only'
SAVE_ROOT = '/home/dmsl/nas/DMSL/AFEW/NpzData/Train_only'
X_DATA = SAVE_ROOT + '/x_train_only_frames_overlap.npz'
Y_DATA = SAVE_ROOT + '/y_train_only_frames_overlap.npz'

LABEL = {
    'angry': '/Angry',
    'disgust': '/Disgust',
    'fear': '/Fear',
    'happy': '/Happy',
    'neutral': '/Neutral',
    'sad': '/Sad',
    'surprise': '/Surprise'
}

# TODO : choose overlap data
OVERLAP = True
GATHER_FRAME = 16
x_features = []
y = []
frames = []
count = 0
for key, label in sorted(LABEL.items()):
    y_label = key2label(key)

    label_path = DATA_ROOT + label
    videos = os.listdir(label_path)
    print("Preprocessing label : {}".format(key))
    for j, video in enumerate(videos):
        video_path = label_path + '/' + video + '/'
        image_paths = os.listdir(video_path)
        numFrame = len(image_paths)
        print("Step {j} - Label : {key}, Video : {video}, Total frames : {numFrame}".format(
            j=j,
            key=video_path.split('/')[-3],
            video=video_path.split('/')[-2],
            numFrame=numFrame))

        if numFrame < GATHER_FRAME:
            numpad = GATHER_FRAME - numFrame
            for image_path in sorted(image_paths):
                image_path = video_path + image_path
                img = preprocess_img(image_path)

                feature = feature_extract_model.predict(img)
                frames.append(feature)
            for _ in range(numpad):
                frames.append(np.zeros((1, 4096)))

            y.append([y_label])
            x_features.append(check16Frames(frames, GATHER_FRAME))
            frames = []
            count += 1

        else:
            if not OVERLAP:
                for i, image_path in enumerate(sorted(image_paths)):
                    if i >= GATHER_FRAME:
                        break
                    image_path = video_path + image_path
                    img = preprocess_img(image_path)

                    feature = feature_extract_model.predict(img)
                    frames.append(feature)
                y.append([y_label])
                x_features.append(check16Frames(frames, GATHER_FRAME))
                frames = []
                count += 1
            else:
                for i_ in range(numFrame - GATHER_FRAME + 1):
                    for j_ in range(i_, GATHER_FRAME + i_):
                        image_path = video_path + image_paths[j_]
                        img = preprocess_img(image_path)

                        feature = feature_extract_model.predict(img)
                        frames.append(feature)

                    y.append([y_label])
                    x_features.append(check16Frames(frames, GATHER_FRAME))
                    frames = []
                    count += 1

                print("# of overlap data : {}".format(numFrame - GATHER_FRAME + 1))


print("Total data : {}".format(count))
x_features = np.vstack(x_features)
y = np.vstack(y)

np.savez(X_DATA, x_features)
np.savez(Y_DATA, y)
