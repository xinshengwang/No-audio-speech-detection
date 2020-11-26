# No-audio-speech-detection

No-audio Multimodal Speech Detection is one of the tasks in MediaEval 2020, with the goal to automatically detect whether someone is speaking in social interaction on the basis of body movement signals. In this code, a multimodal fusion method, combining signals obtained by an overhead camera and a wearable accelerometer, was proposed to determine whether someone was speaking. The proposed system directly takes the accelerometer signals as input, while using a pre-trained 3D convolutional network to extract the video features that work as input. 

### Requirements

* python 3.6
* pytorch 1.4.0

### Run the code
for accelerometer
```
python acl.py 
```

for video
```
python c3d.py
```

for fusion

```
python train.py
```

### Cite

@article{wang2020multimodal ,
title={Multimodal Fusion of Body Movement Signals for No-audio Speech Detection},
author={Xinsheng Wang, Jihua Zhu, Odette Scharenborg},
journal={MediaEval'20},
year={2020}
}

