# Emotion Recognition in the Wild Challenge (EmotiW 2018) audio-video sub-challenge

### Description
 The audio-video sub-challenge focuses on emotion classification tasks based on videos, 
which contains seven basic emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral. \
For more detailed information, please refer to webisite [[EmotiW 2018]](https://sites.google.com/view/emotiw2018). 
Finally, we got **59.72%** accuracy in this challenge and we are **7th** in 32 teams. It can be seen [this link](https://arxiv.org/abs/1808.07773). 

We use AFEW dataset that use EmotiW challenge. You can get AFEW dataset after the administrator's approval in EmotiW website.

Our method is combinations of 3D CNN, 2D CNN, and RNN.
- 3D CNN
  - C3D
  - ResNet 3D
  - ResNeXt 3D 
- 2D CNN
  - DenseNet
- RNN
  - LSTM
  - GRU


**Note**: This repository can get accuracy for validation set. Because there is no label for the testset of AFEW dataset, above accuracy is not available.

### Environment
This code is tested on below setting.
- Ubuntu 16.04
- Python 3.5
- Cuda 9.0 
- [Pytorch](https://pytorch.org)

### Usage
**Step 1. Clone this respository to local.**
```angular2html
git clone https://github.com/lemin/EmotiW-2018.git
cd EmotiW-2018
```

**Step 2. Prepare dataset.** \
1. Pre-processing method is to crop the face after detecting the face every frame and make it into `.npz` file.
The algorithm we used is [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment). 
2. We extract 4069-d feature for every data frame by frame using DenseNet after training DenseNet. 
Likewise, save these features as `.npz` file.

**Step 3. Train networks.**
```angular2html
python main.py
```

