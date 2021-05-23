# MultiStage-ActionDetection

This repository contains the inference code of a multi-stage action detection pipeline based on the [Object_Detection_Tracking](https://github.com/JunweiLiang/Object_Detection_Tracking) repository and the [Video_Classification](https://github.com/HHTseng/video-classification) repository. The pipeline is composed of three major components:

1. Object Detection module
2. Object Tracking module
3. Action Classification module

The multi-batch and multi-thread inference has been performed on the [MEVA](https://mevadata.org/) dataset, a 1920x1080 resolution RGB video sequence comprised of Person and Vehicle instances.

## Introduction

For the object detection module, we use MaskRCNN with a Resnet50 + FPN backbone pre-trained on the MSCOCO dataset from the [Detectron2](https://github.com/facebookresearch/detectron2) library. As for the object tracking module, we use the DeepSORT implementation of the [Object_Detection_Tracking](https://github.com/JunweiLiang/Object_Detection_Tracking) repository by adapting it for the PyTorch version of our pipeline. For the activity classification, we use a Convolutional RNN model from the [Video_Classification](https://github.com/HHTseng/video-classification) repository which uses a pre-trained Resnet CNN for the spatial information and an LSTM network for the temporal information.

## Installation

Requirements:

- Python>=3.7
- cuda10.1
- PyTorch=1.5.1
- Torchvision=0.6.1
- Pillow=7.0.0

Other dependencies: numpy; scipy; sklearn; cv2; matplotlib; pycocotools

This inferencing code is tested with on Google Colaboratory GPU and Python 3.

## Code Overview

- `inference.py`: Inference code for the multi-stage action detection.
- `vis_json.py`: Code for visualizing the json outputs.
- `get_frames_resize.py`: Code for extracting frames from videos.
- `single_video_reid.py`: To generated ReID output representation.

## Inferencing

1. First download some test videos:

```
$ wget https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/meva_outdoor_test.tgz
$ tar -zxvf meva_outdoor_test.tgz
$ ls meva_outdoor_test > meva_outdoor_test.lst
```

2. Run object detection & tracking on the test videos with batch_size=8 code:

```
$ python inference_multibatch_multithread.py --video_dir meva_outdoor_test --video_lst_file meva_outdoor_test.lst --frame_gap 8 \
--get_tracking --tracking_dir fpnr50_multib8thread_trackout_1280x720 --max_size 1280 --short_edge_size 800 --is_coco --im_batch_size 8 --prefetch 10
```

3. You can visualize the results according to instructions of the [Object_Detection_Tracking](https://github.com/JunweiLiang/Object_Detection_Tracking) repository.
