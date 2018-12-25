import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset

import os
import numpy as np
from util import Util as util

ROOT = '/home/dmsl/nas/HDD_server6/EmotiW/data/'
VIDEO_TRAIN_PATH = ROOT + 'VidNpz/Train_only'
VIDEO_VAL_PATH = ROOT + 'VidNpz/Val'
VIDEO_TEST_PATH = ROOT + 'VidNpz/newTest2018_3rd'
# VIDEO_TEST_PATH = ROOT + 'VidNpz/Test2017'
VIDEO_EXTERNAL1_PATH = ROOT + 'VidNpz/External_LHW'
VIDEO_EXTERNAL2_PATH = ROOT + 'VidNpz/External_KDH'
VIDEO_EXTERNAL2017_PATH = ROOT + 'VidNpz/Train_add'

IMAGE_TRAIN_PATH = ROOT + 'ImgNpz/Train_only'
IMAGE_VAL_PATH = ROOT + 'ImgNpz/Val'
IMAGE_TEST_PATH = ROOT + 'ImgNpz/newTest2018_3rd'
# IMAGE_TEST_PATH = ROOT + 'ImgNpz/Test2017'
IMAGE_EXTERNAL1_PATH = ROOT + 'ImgNpz/External_LHW'
IMAGE_EXTERNAL2_PATH = ROOT + 'ImgNpz/External_KDH'
IMAGE_EXTERNAL2017_PATH = ROOT + 'ImgNpz/Train_add'

def get_loader(batch_size, num_workers):
    # TODO : compact script
    X_VIDEO_TRAIN_PATH = os.path.join(VIDEO_TRAIN_PATH, 'x_afew_rchoice_flip_no_overlap.npz')
    # X_VIDEO_VAL_PATH = os.path.join(VIDEO_VAL_PATH, 'x_afew_rchoice_flip_no_overlap.npz') # Train = Train_only + validation
    X_VIDEO_VAL_PATH = os.path.join(VIDEO_VAL_PATH, 'x_afew_rchoice_no_overlap.npz') # Validation mode
    #X_VIDEO_TEST_PATH = os.path.join(VIDEO_TEST_PATH, 'x_afew_rchoice_no_overlap.npz')
    #X_VIDEO_EXTERNAL1_PATH = os.path.join(VIDEO_EXTERNAL1_PATH, 'x_afew_rchoice_flip_no_overlap.npz')
    #X_VIDEO_EXTERNAL2_PATH = os.path.join(VIDEO_EXTERNAL2_PATH, 'x_afew_rchoice_flip_no_overlap.npz')
    # X_VIDEO_EXTERNAL2017_PATH = os.path.join(VIDEO_EXTERNAL2017_PATH, 'x_afew_rchoice_flip_no_overlap.npz')


    X_IMAGE_TRAIN_PATH = os.path.join(IMAGE_TRAIN_PATH, 'x_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    Y_IMAGE_TRAIN_PATH = os.path.join(IMAGE_TRAIN_PATH, 'y_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    # X_IMAGE_VAL_PATH = os.path.join(IMAGE_VAL_PATH, 'x_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz') # Train = Train_only + validation
    # Y_IMAGE_VAL_PATH = os.path.join(IMAGE_VAL_PATH, 'y_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz') # Train = Train_only + validation
    X_IMAGE_VAL_PATH = os.path.join(IMAGE_VAL_PATH, 'x_DenseNet121_titu_fc2_fc1_no_overlap.npz') # Validation mode
    Y_IMAGE_VAL_PATH = os.path.join(IMAGE_VAL_PATH, 'y_DenseNet121_titu_fc2_fc1_no_overlap.npz') # Validation mode
    #X_IMAGE_TEST_PATH = os.path.join(IMAGE_TEST_PATH, 'x_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    #Y_IMAGE_TEST_PATH = os.path.join(IMAGE_TEST_PATH, 'y_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    #X_IMAGE_EXTERNAL1_PATH = os.path.join(IMAGE_EXTERNAL1_PATH, 'x_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    #Y_IMAGE_EXTERNAL1_PATH = os.path.join(IMAGE_EXTERNAL1_PATH, 'y_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    #X_IMAGE_EXTERNAL2_PATH = os.path.join(IMAGE_EXTERNAL2_PATH, 'x_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    #Y_IMAGE_EXTERNAL2_PATH = os.path.join(IMAGE_EXTERNAL2_PATH, 'y_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    # X_IMAGE_EXTERNAL2017_PATH = os.path.join(IMAGE_EXTERNAL2017_PATH, 'x_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz')
    # Y_IMAGE_EXTERNAL2017_PATH = os.path.join(IMAGE_EXTERNAL2017_PATH, 'y_flip_DenseNet121_titu_fc2_fc1_no_overlap.npz')

    x_video_train, x_video_val = util.load_from_npz(X_VIDEO_TRAIN_PATH), util.load_from_npz(X_VIDEO_VAL_PATH)
    x_image_train, x_image_val = util.load_from_npz(X_IMAGE_TRAIN_PATH), util.load_from_npz(X_IMAGE_VAL_PATH)
    y_image_train, y_image_val = util.load_from_npz(Y_IMAGE_TRAIN_PATH), util.load_from_npz(Y_IMAGE_VAL_PATH)
    #x_video_test = util.load_from_npz(X_VIDEO_TEST_PATH)
    #x_image_test, y_image_test = util.load_from_npz(X_IMAGE_TEST_PATH), util.load_from_npz(Y_IMAGE_TEST_PATH)
    #x_video_external1 = util.load_from_npz(X_VIDEO_EXTERNAL1_PATH)
    #x_image_external1, y_image_external1 = util.load_from_npz(X_IMAGE_EXTERNAL1_PATH), util.load_from_npz(Y_IMAGE_EXTERNAL1_PATH)
    #x_video_external2 = util.load_from_npz(X_VIDEO_EXTERNAL2_PATH)
    #x_image_external2, y_image_external2 = util.load_from_npz(X_IMAGE_EXTERNAL2_PATH), util.load_from_npz(Y_IMAGE_EXTERNAL2_PATH)
    # x_video_external2017= util.load_from_npz(X_VIDEO_EXTERNAL2017_PATH)
    # x_image_external2017, y_image_external2017 = util.load_from_npz(X_IMAGE_EXTERNAL2017_PATH), util.load_from_npz(Y_IMAGE_EXTERNAL2017_PATH)

    #x_video_train = np.vstack([x_video_train, x_video_external1, x_video_external2])#, x_video_external1, x_video_external2]) # Train = Train_only + validation
    #x_image_train = np.vstack([x_image_train, x_image_external1, x_image_external2])#, x_image_external1, x_image_external2]) # Train = Train_only + validation
    #y_image_train = np.vstack([y_image_train, y_image_external1, y_image_external2])#, y_image_external1, y_image_external2]) # Train = Train_only + validation

    train_datasets = VideoNImageDataset(x_video_train, x_image_train, y_image_train)
    val_datasets = VideoNImageDataset(x_video_val, x_image_val, y_image_val) # Validation mode
    # test_datasets = VideoNImageDataset(x_video_test, x_image_test, y_image_test)

    # train & validation & test
    # dataloaders = {
    #     'train': DataLoader(
    #         dataset=train_datasets,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         collate_fn=collate_fn,
    #         num_workers=num_workers),
    #     'val': DataLoader(
    #         dataset=val_datasets,
    #         batch_size=batch_size,
    #         collate_fn=collate_fn,
    #         num_workers=num_workers),
    #     'test': DataLoader(
    #         dataset=test_datasets,
    #         batch_size=batch_size,
    #         collate_fn=collate_fn,
    #         num_workers=num_workers
    #     )
    # }
    # dataset_sizes = {
    #     'train': x_video_train.shape[0],
    #     'val': x_video_val.shape[0],
    #     'test': x_video_test.shape[0]
    # }

    # Train = Train_only + validation
    # dataloaders = {
    #     'train': DataLoader(
    #         dataset=train_datasets,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         collate_fn=collate_fn,
    #         num_workers=num_workers),
    #     'test': DataLoader(
    #         dataset=test_datasets,
    #         batch_size=batch_size,
    #         collate_fn=collate_fn,
    #         num_workers=num_workers
    #     )
    # }
    # dataset_sizes = {
    #     'train': x_video_train.shape[0],
    #     'test': x_video_test.shape[0]
    # }

    # validation
    dataloaders = {
        'train': DataLoader(
            dataset=train_datasets,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers),
        'test': DataLoader(
            dataset=val_datasets,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    }
    dataset_sizes = {
        'train': x_video_train.shape[0],
        'test': x_video_val.shape[0]
    }

    print('# of train data : {} \n# of test data : {}'.format(dataset_sizes['train'], dataset_sizes['test']))
    # return dataloaders, dataset_sizes, y_image_test
    return dataloaders, dataset_sizes, y_image_val

class VideoNImageDataset(Dataset):
    def __init__(self, video_data, image_data, image_target):
        self.video_data = video_data.transpose(0, 4, 1, 2, 3)
        self.image_data = image_data
        self.image_target = image_target

    def __len__(self):
        return self.video_data.shape[0]

    def __getitem__(self, idx):
        video = torch.from_numpy(self.video_data[idx])
        image = torch.Tensor(self.image_data[idx])
        i_target = torch.LongTensor(self.image_target[idx])
        return video, image, i_target

def collate_fn(data):
    video_data, image_data, image_target = zip(*data)
    video_data = torch.stack(video_data, dim=0)
    image_data = torch.stack(image_data, dim=0)
    image_target = torch.stack(image_target, dim=0)

    return video_data, image_data, image_target.squeeze(1)
