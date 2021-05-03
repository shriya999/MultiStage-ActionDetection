# coding=utf-8
"""
  run object detection and tracking inference
"""

import argparse
import cv2
import math
import json
import random
import sys
import time
import threading
import operator
import os
import pickle
from enqueuer_thread import VideoEnqueuer
import matplotlib
# avoid the warning "gdk_cursor_new_for_display:
# assertion 'GDK_IS_DISPLAY (display)' failed" with Python 3
matplotlib.use('Agg')
from tqdm import tqdm
import torch, torchvision
import numpy as np

# detection stuff
from utils import get_op_tensor_name
from utils import parse_nvidia_smi
from utils import sec2time
from utils import PerformanceLogger
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# tracking stuff
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing
from deep_sort.utils import create_obj_infos,linear_inter_bbox,filter_short_objs

# for mask
import pycocotools.mask as cocomask

# class ids stuff
from class_ids import targetClass2id_new_nopo
from class_ids import coco_obj_class_to_id
from class_ids import coco_obj_id_to_class
from class_ids import coco_obj_to_actev_obj
from class_ids import coco_id_mapping

targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id[one]: one for one in targetClass2id}


def get_args():
    """Parse arguments and intialize some hyper-params."""
    global targetClass2id, targetid2class
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--video_lst_file", default=None,
                        help="video_file_path = os.path.join(video_dir, $line)")
    parser.add_argument("--frame_gap", default=8, type=int)

    parser.add_argument("--is_coco_model", action="store_true",
                      help="is coco model, will output coco classes instead")

    parser.add_argument("--max_size", type=int, default=1920,
                      help="num roi per image for RPN and fastRCNN training")
    parser.add_argument("--short_edge_size", type=int, default=1080,
                      help="num roi per image for RPN and fastRCNN training")

    # ---- gpu params
    parser.add_argument("--gpu", default=1, type=int, help="number of gpu")
    parser.add_argument("--gpuid_start", default=0, type=int,
                        help="start of gpu id")
    parser.add_argument("--im_batch_size", type=int, default=1)

    # ----------- model params
    parser.add_argument("--num_class", type=int, default=15,
                        help="num catagory + 1 background")

    # ----------- tracking params
    parser.add_argument("--tracking_objs", default="Person,Vehicle",
                        help="Objects to be tracked, default are Person and "
                            "Vehicle")
    parser.add_argument("--get_tracking", action="store_true",
                      help="this will generate tracking results for each frame")
    parser.add_argument("--tracking_dir", default="/tmp",
                      help="output will be out_dir/$videoname.txt, start from 0"
                           " index")

    # ---- for multi-thread frame preprocessing
    parser.add_argument("--prefetch", type=int, default=10,
                        help="maximum number of batch in queue")

    args = parser.parse_args()
    targetid2class = targetid2class
    targetClass2id = targetClass2id

    targetClass2id = coco_obj_class_to_id
    targetid2class = coco_obj_id_to_class
    args.num_class = 81
    partial_classes = ["BG"] + args.partial_classes
    targetClass2id = {classname: i
                    for i, classname in enumerate(partial_classes)}
    targetid2class = {targetClass2id[o]: o for o in targetClass2id}

    args.classname2id = targetClass2id
    args.classid2name = targetid2class

    # ---------------more defautls
    args.diva_class3 = True
    args.diva_class = False
    args.diva_class2 = False
    args.use_small_object_head = False
    args.use_so_score_thres = False
    args.max_size = np.ceil(args.max_size / args.fpn_resolution_requirement) * \
                    args.fpn_resolution_requirement
    args.result_score_thres = args.threshold_conf
    args.result_per_im = 100

    return args

def check_args(args):
    """Check the argument."""
    assert args.video_dir is not None
    assert args.video_lst_file is not None
    assert args.frame_gap >= 1
    #print("cv2 version %s" % (cv2.__version__)


def run_detect_and_track(args, frame_stack, model, targetid2class,
                         tracking_objs,
                         tracker_dict, tracking_results_dict,
                         tmp_tracking_results_dict,
                         obj_out_dir=None,
                         valid_frame_num=None):
    # ignore the padded images
    if valid_frame_num is None:
        valid_frame_num = len(frame_stack)

    resized_images, scales, frame_idxs = zip(*frame_stack)

    # [B, H, W, 3] 
    batched_imgs = np.stack(resized_images, axis=0)
    # [B, H, W, 3] -> [B, C, H, W] for pytorch
    reordered_imgs = np.moveaxis(batched_images, (0, 1, 2, 3), (0, 3, 1, 2))
    # [B, num, 4], [B, num], [B, num], [B], [M, 256, 7, 7]
    batch_boxes, batch_labels, batch_probs, valid_indices, batch_box_feats = inference_detector(model, reordered_imgs)
    assert np.sum(valid_indices) == batch_box_feats.shape[0], "duh"

    for b in range(valid_frame_num):
      cur_frame = frame_idxs[b]
      final_boxes = batch_boxes[b][:valid_indices[b]] # [k, 4]
      final_labels = batch_labels[b][:valid_indices[b]] # [k]
      final_probs = batch_probs[b][:valid_indices[b]]  # [k]
      previous_box_num = sum(valid_indices[:b])  # [k, 256, 7, 7]
      box_feats = batch_box_feats[previous_box_num:previous_box_num+valid_indices[b]]

      if args.get_tracking:
        assert len(box_feats) == len(final_boxes)

        for tracking_obj in tracking_objs:
          target_tracking_obs = [tracking_obj]
          # will consider scale here
          scale = scales[b]
          detections = create_obj_infos(
              cur_frame, final_boxes, final_probs, final_labels, box_feats,
              targetid2class, target_tracking_obs, args.min_confidence,
              args.min_detection_height, scale,
              is_coco_model=args.is_coco_model,
              coco_to_actev_mapping=coco_obj_to_actev_obj)
          # Run non-maxima suppression.
          boxes = np.array([d.tlwh for d in detections])
          scores = np.array([d.confidence for d in detections])
          indices = preprocessing.non_max_suppression(
              boxes, 0.85, scores)
          detections = [detections[i] for i in indices]

          # tracking
          tracker_dict[tracking_obj].predict()
          tracker_dict[tracking_obj].update(detections)

          # Store results
          for track in tracker_dict[tracking_obj].tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
              if (not track.is_confirmed()) and track.time_since_update == 0:
                bbox = track.to_tlwh()
                if track.track_id not in \
                    tmp_tracking_results_dict[tracking_obj]:
                  tmp_tracking_results_dict[tracking_obj][track.track_id] = \
                      [[cur_frame, track.track_id, bbox[0], bbox[1],
                        bbox[2], bbox[3]]]
                else:
                  tmp_tracking_results_dict[
                      tracking_obj][track.track_id].append(
                          [cur_frame, track.track_id,
                          bbox[0], bbox[1], bbox[2], bbox[3]])
              continue
            bbox = track.to_tlwh()
            if track.track_id in tmp_tracking_results_dict[tracking_obj]:
              pred_list = tmp_tracking_results_dict[tracking_obj][
                  track.track_id]
              for pred_data in pred_list:
                tracking_results_dict[tracking_obj].append(pred_data)
              tmp_tracking_results_dict[tracking_obj].pop(track.track_id,
                                                          None)
            tracking_results_dict[tracking_obj].append([cur_frame, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

      if obj_out_dir is None:  # not saving the boxes
        continue

      # ---------------- get the json outputs for object detection

      # scale back the box to original image size
      final_boxes = final_boxes / scales[b]

      # save as json
      pred = []
      for j, (box, prob, label) in enumerate(zip(
          final_boxes, final_probs, final_labels)):
        box[2] -= box[0]
        box[3] -= box[1]  # produce x,y,w,h output

        cat_id = int(label)
        cat_name = targetid2class[cat_id]

        res = {
            "category_id": int(cat_id),
            "cat_name": cat_name,  # [0-80]
            "score": float(round(prob, 7)),
            #"bbox": list(map(lambda x: float(round(x, 2)), box)),
            "bbox": [float(round(x, 2)) for x in box],
            "segmentation": None,
        }
        pred.append(res)
      predfile = os.path.join(obj_out_dir, "%d.json" % (cur_frame))

      with open(predfile, "w") as f:
        json.dump(pred, f)

if __name__ == "__main__":
  args = get_args()
  check_args(args)

  videolst = [os.path.join(args.video_dir, one.strip())
              for one in open(args.video_lst_file).readlines()]

  from diva_io.video import VideoReader

  # 1. load the object detection model
  config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'
  checkpoint = 'checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
  model = init_detector(config, checkpoint, device='cuda:0')

  for videofile in tqdm(videolst, ascii=True):
    # 2. read the video file
    vcap = VideoReader(videofile)
    frame_count = int(vcap.length)

    # initialize tracking module
    if args.get_tracking:
      tracking_objs = args.tracking_objs.split(",")
      tracker_dict = {}
      tracking_results_dict = {}
      tmp_tracking_results_dict = {}
      for tracking_obj in tracking_objs:
        metric = metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", args.max_cosine_distance, args.nn_budget)
        tracker_dict[tracking_obj] = Tracker(
            metric, max_iou_distance=args.max_iou_distance)
        tracking_results_dict[tracking_obj] = []
        tmp_tracking_results_dict[tracking_obj] = {}

    # videoname = os.path.splitext(os.path.basename(videofile))[0]
    videoname = os.path.basename(videofile)
    video_obj_out_path = None
    video_queuer = VideoEnqueuer(
        args, vcap, frame_count, frame_gap=args.frame_gap,
        prefetch=args.prefetch,
        start=True, is_moviepy=args.use_moviepy,
        batch_size=args.im_batch_size)
    get_batches = video_queuer.get()

    for batch in tqdm(get_batches, total=video_queuer.num_batches):
      # batch is a list of (resized_image, scale, frame_count)
      valid_frame_num = len(batch)
      if len(batch) < args.im_batch_size:
        batch += [batch[-1]] * (args.im_batch_size - len(batch))

      run_detect_and_track(args, batch, model, targetid2class, tracking_objs, tracker_dict, tracking_results_dict,
          tmp_tracking_results_dict, video_obj_out_path, valid_frame_num=valid_frame_num)

    if args.get_tracking:
      for tracking_obj in tracking_objs:
        output_dir = os.path.join(args.tracking_dir, videoname, tracking_obj)
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "%s.txt" % (os.path.splitext(videoname))[0])

        tracking_results = sorted(tracking_results_dict[tracking_obj], key=lambda x: (x[0], x[1]))
        # print(len(tracking_results)
        tracking_data = np.asarray(tracking_results)
        # print(tracking_data.shape
        tracking_data = linear_inter_bbox(tracking_data, args.frame_gap)
        tracking_data = filter_short_objs(tracking_data)
        tracking_results = tracking_data.tolist()
        with open(output_file, "w") as fw:
          for row in tracking_results:
            line = "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" % (
                row[0], row[1], row[2], row[3], row[4], row[5])
            fw.write(line + "\n")