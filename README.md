# MultiStage-ActionDetection

This repository contains the inference code of a multi-stage action detection pipeline based on [this](https://github.com/JunweiLiang/Object_Detection_Tracking) repository. The pipeline is composed of three major components:

1. Object Detection module (Done)
2. Object Tracking module (Done)
3. Action Classification module (To Do)

The multi-batch and multi-thread inference has been performed on the [MEVA](https://mevadata.org/) dataset, a 1920x1080 resolution RGB video sequence comprised of Person and Vehicle instances.

## Introduction

For the object detection module, we use the MaskRCNN with a Resnet50 + FPN backbone pre-trained on the MSCOCO dataset from the [Detectron2](https://github.com/facebookresearch/detectron2) library. As for the object tracking module, we use the DeepSORT implementation of [this](https://github.com/JunweiLiang/Object_Detection_Tracking) repository by adapting it for the PyTorch version of our pipeline.

## Installation

Requirements:

Python>=3.7
cuda10.1
PyTorch=1.5.1
Torchvision=0.6.1
Pillow=7.0.0

Other dependencies: numpy; scipy; sklearn; cv2; matplotlib; pycocotools

This inferencing code is tested with on Google Colaboratory GPU and Python 3.

## Code Overview

- `inference_multibatch_multithread.py`: Inference code for object detection & tracking.
- `vis_json.py`: visualize the json outputs.
- `get_frames_resize.py`: code for extracting frames from videos.
- `utils.py`: some helper classes 

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

3. You can visualize the results according to instructions of [this](https://github.com/JunweiLiang/Object_Detection_Tracking) repository.
