# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from class_ids import coco_obj_class_to_id
from class_ids import coco_obj_id_to_class
from class_ids import coco_obj_to_actev_obj

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class RCNNPredictor(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 300,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        super().__init__(
            input_shape=input_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=test_score_thresh,
            test_nms_thresh=test_nms_thresh,
            test_topk_per_image=test_topk_per_image,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            loss_weight=loss_weight,
        )

    def inference(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        return self.box_pred_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def box_pred_inference(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
    ):
        """
        Call `fast_rcnn_inference_single_image` for all images.

        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
                all detections.

        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        result_per_image = [
            self.box_pred_inference_single_image(
                boxes_per_image,
                scores_per_image,
                image_shape,
                score_thresh,
                nms_thresh,
                topk_per_image,
            )
            for scores_per_image, boxes_per_image, image_shape in zip(
                scores, boxes, image_shapes
            )
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def box_pred_inference_single_image(
        self,
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Args:
            Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
            per image.

        Returns:
            Same as `fast_rcnn_inference`, but for only one image.
        """

        partial_classes = [classname for classname in coco_obj_to_actev_obj]
        classname2id = coco_obj_class_to_id
        partial_classes = [classname for classname in coco_obj_to_actev_obj]
        needed_object_classids = [classname2id[name] for name in partial_classes]
        needed_object_classids_minus_1 = [o - 1 for o in needed_object_classids]

        # (N, num_class), (N, num_class - 1, 4)
        # -> (num_class, N), (num_class - 1, N, 4)
        box_logits = boxes.reshape(boxes.shape[0], boxes.shape[1] // 4, 4)
        label_logits_t = scores.permute(1, 0)
        box_logits_t = box_logits.permute(1, 0, 2)
        # [C + 1, N]  # 1 is the BG class
        partial_label_logits_t = label_logits_t[[0] + needed_object_classids]
        # [C, N, 4]
        partial_box_logits_t = box_logits_t[needed_object_classids_minus_1]

        partial_label_logits = partial_label_logits_t.permute(1, 0)
        partial_box_logits = partial_box_logits_t.permute(1, 0, 2)
        scores = partial_label_logits
        boxes = partial_box_logits.reshape(partial_box_logits.shape[0], -1)

        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(
            dim=1
        )
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]

        scores = scores[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        filter_mask = scores > 0.0  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        return result, filter_inds[:, 0]
