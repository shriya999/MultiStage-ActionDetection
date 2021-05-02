import torch, torchvision
import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# Setup a checkpoint file to load
checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# initialize the mmdetector
model = init_detector(config, checkpoint, device='cuda:0')

# Use the detector to do inference
img = 'demo/demo.jpg'
result = inference_detector(model, img)

# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.3)