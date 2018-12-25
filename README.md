# Emotion Recognition in the Wild Challenge (EmotiW 2018) audio-video sub-challenge

### Description
 The audio-video sub-challenge focuses on emotion classification tasks based on videos, 
which contains seven basic emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral. \
For more detailed information, please refer to webisite [[EmotiW 2018]](https://sites.google.com/view/emotiw2018). 
Finally, we got **59.72%** accuracy in this challenge and we are **7th** in 32 teams. It can be seen [this link](https://arxiv.org/abs/1808.07773). 

We use AFEW dataset that use EmotiW challenge. You can get AFEW dataset after the administrator's approval in EmotiW website.

**Note**: This repository can get accuracy for validation set. Because there is no label for the testset of AFEW dataset, above accuracy is not available.

### Environment
This code is tested on below setting.
- Ubuntu 16.04
- Python 3.5
- Cuda 9.0 
- [Pytorch](https://pytorch.org) 0.4.1

### Usage
**Step 1. Clone this respository to local.**
```angular2html
git clone https://github.com/lemin/emotiw.git
cd emotiw
```

**Step 2. Prepare dataset.** \
Pre-processing method is to crop the face after detecting the face every frame and make it into `.npz` file.
The algorithm we used is [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment).

**Step 3. Train networks.**
```angular2html
python main.py
```

