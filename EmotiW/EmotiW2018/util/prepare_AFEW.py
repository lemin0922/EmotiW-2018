import os
from skimage.measure import block_reduce
import cv2
import numpy as np
from six.moves import xrange
import matplotlib
#matplotlib.use('qt5agg')
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from collections import defaultdict


def crop_and_resize(img, target_size=32, zoom=1):
    small_side = int(np.min(img.shape) * zoom)
    reduce_factor = small_side / target_size
    crop_size = target_size * reduce_factor
    mid = np.array(img.shape) / 2
    half_crop = crop_size / 2
    center = img[mid[0] - half_crop:mid[0] + half_crop,
             mid[1] - half_crop:mid[1] + half_crop]
    return block_reduce(center, (reduce_factor, reduce_factor), np.mean)


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


def to_dataset(examples):
    X = []
    y = []
    for path, label in examples:
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) # ImageNet Size
        X.append(img)
        y.append(label)

    return np.asarray(X), np.asarray(y)

if __name__ == '__main__':
    # TODO : confirm here!
    DATA_ROOT = '/home/dmsl/nas/DMSL/AFEW/afew-mtcnn/REARRANGE/Train_only'
    SAVE_ROOT = '/home/dmsl/nas/DMSL/AFEW/NpzData/Train_only'
    X_DATA = '/x_val_mtcnn_color.npz'
    Y_DATA = '/y_val_mtcnn_color.npz'

    LABEL = {
        'angry': '/Angry',
        'disgust': '/Disgust',
        'fear': '/Fear',
        'happy': '/Happy',
        'neutral': '/Neutral',
        'sad': '/Sad',
        'surprise': '/Surprise'
    }
    paths_dict = defaultdict(lambda : [])
    label_dict = defaultdict(lambda : [])

    num = 0
    for key, value in sorted(LABEL.items()):
        label_path = DATA_ROOT + value
        box = list(list_box(label_path, ['.jpg', '.png']))
        paths = box_paths(label_path, box)
        temp_num = 0
        for path in paths:
            temp = list(list_all_files(path, ['.jpg', '.png']))
            paths_dict[key] += temp
            num += len(temp)
            temp_num += len(temp)
        print("Loaded", temp_num, key + ' file lists')
    print('Total # of paths : {}'.format(num))

    for key in sorted(list(paths_dict.keys())):
        if key == 'angry':
            label = 0
        elif key == 'disgust':
            label = 1
        elif key == 'fear':
            label = 2
        elif key == 'happy':
            label = 3
        elif key == 'neutral':
            label = 4
        elif key == 'sad':
            label = 5
        elif key == 'surprise':
            label = 6

        for path in paths_dict[key]:
            label_dict[key] += [(path, label)]

    X_data_list = []
    Y_data_list = []
    for key in label_dict.keys():
        X, Y = to_dataset(label_dict[key])
        Y = Y.reshape((Y.shape[0], 1))
        X_data_list.append(X)
        Y_data_list.append(Y)

    X_data = np.vstack(X_data_list)
    y_data = np.vstack(Y_data_list)

    np.savez(SAVE_ROOT + X_DATA, X_data)
    np.savez(SAVE_ROOT + Y_DATA, y_data)

    print("Save all data!")
