import cv2
import numpy as np
import torch

from os2d.modeling.model import build_os2d_from_config
from os2d.config import cfg
from os2d.structures.feature_map import FeatureMapSize
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio
from torchvision.transforms.functional import to_tensor, normalize


class OneShotDetector:

    def __init__(self, cfg_path:str, score_threshold:float = 0.5):
        cfg.merge_from_file(cfg_path)
        cfg.is_cuda = torch.cuda.is_available()
        cfg.freeze()
        self.net, self.box_coder, criterion, self.img_normalization, optimizer_state = build_os2d_from_config(cfg)

        self.source_img_size = 1500
        self.max_detections = 1
        self.score_threshold = score_threshold

    def _preprocess(self, img:np.ndarray, size:int):
        h, w = get_image_size_after_resize_preserving_aspect_ratio(h=img.shape[0],
                                                                   w=img.shape[1],
                                                                   target_size=size)
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_NEAREST)

        img = to_tensor(img)
        img = normalize(img, self.img_normalization["mean"], self.img_normalization["std"])
        img = img.unsqueeze(0)
        if cfg.is_cuda:
            img = img.cuda()

        return img

    def detect(self, target, source):
        target = self._preprocess(target, cfg.model.class_image_size)
        source = self._preprocess(source, self.source_img_size)

        with torch.no_grad():
            loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = \
                self.net(images=source, class_images=target)

        image_loc_scores_pyramid = [loc_prediction_batch[0]]
        image_class_scores_pyramid = [class_prediction_batch[0]]
        img_size_pyramid = [FeatureMapSize(img=source)]
        transform_corners_pyramid = [transform_corners_batch[0]]

        class_ids = [0]
        boxes = self.box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                         img_size_pyramid, class_ids,
                                         nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                         nms_score_threshold=cfg.eval.nms_score_threshold,
                                         transform_corners_pyramid=transform_corners_pyramid)
        boxes.remove_field("default_boxes")

        scores = boxes.get_field("scores")

        good_ids = torch.nonzero(scores.float() > self.score_threshold).view(-1)
        if good_ids.numel() > 0:
            _, ids = scores[good_ids].sort(descending=False)
            good_ids = good_ids[ids[-self.max_detections:]]
            boxes = boxes[good_ids].cpu()
            scores = scores[good_ids].cpu()
            boxes = boxes.bbox_xyxy
            boxes[:, [0,2]] /= source.shape[3]
            boxes[:, [1,3]] /= source.shape[2]
            return boxes, scores
        else:
            return None, None