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
    for key, label in sorted(LABEL.items()):
        label_path = DATA_ROOT + label
        videos = os.listdir(label_path)
        for video in videos:
            video_path = label_path + '/' + video
            image_paths = os.listdir(video_path)
            for image_path in sorted(image_paths):
                path = video_path + image_path
