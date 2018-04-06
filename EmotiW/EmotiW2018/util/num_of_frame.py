import os
import numpy as np
from six.moves import xrange
from skimage.io import imread, imshow
from collections import defaultdict

def list_all_files(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ext.lower() in extensions:
                yield joined


def list_box(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        boxes = []
        boxes.append(dirnames)
        for box in boxes:
            return box


def box_paths(directory, box):
    joined = []
    for i in xrange(len(box)):
        joined.append(os.path.join(directory, str(box[i])))
    return joined

if __name__ == "__main__":
    DATA_ROOT = '/home/dmsl/nas/DMSL/AFEW/afew-mtcnn/frame16/Train_only'
    SAVE_ROOT = '/home/dmsl/nas/DMSL/AFEW/afew-mtcnn/frame16/numOfFrames.txt'

    LABEL = {
        'angry': '/Angry',
        'disgust': '/Disgust',
        'fear': '/Fear',
        'happy': '/Happy',
        'neutral': '/Neutral',
        'sad': '/Sad',
        'surprise': '/Surprise'
    }

    paths = []
    for key, value in LABEL.items():
        data_path = DATA_ROOT + value
        data_box = list(list_box(data_path, ['.jpg', '.png']))
        data_box = box_paths(data_path, data_box)
        paths += data_box

    # with open(SAVE_ROOT, 'w') as f:
    #     for path in paths:
    #         split_path = path.split('/')
    #         label = split_path[-2]
    #         dir_name = split_path[-1]
    #         count = len(os.listdir(path))
    #         f.write(label + ' ' + dir_name + ' ' + str(count) + '\n')

    cnt16 = 0
    for path in paths:
        split_path = path.split('/')
        label = split_path[-2]
        dir_name = split_path[-1]
        count = len(os.listdir(path))
        if count < 8:
            cnt16 += 1
