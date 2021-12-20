import torch
import sys
sys.path.append('pytorch-ssd')

from vision.utils import box_utils
from vision.utils.misc import Timer
from vision.ssd.config import mobilenetv1_ssd_config as config

from transforms import (
    Compose,
    ResizeBatch,
    SubtractMeans,
    ToTensor
)

class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ResizeBatch(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

    def predict_batch(self, batch, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        heights = [image.shape[0] for image in batch]
        widths = [image.shape[1] for image in batch]
        images = self.transform(batch)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            all_scores, all_boxes = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        all_box_coord, all_labels, all_box_probs = [], [], []
        for i, boxes, scores in zip(range(len(heights)),all_boxes, all_scores):
            if not prob_threshold:
                prob_threshold = self.filter_threshold
            # this version of nms is slower on GPU, so we move data to CPU.
            boxes = boxes.to(cpu_device)
            scores = scores.to(cpu_device)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(1, scores.size(1)):
                probs = scores[:, class_index]
                mask = probs > prob_threshold
                probs = probs[mask]
                if probs.size(0) == 0:
                    continue
                subset_boxes = boxes[mask, :]
                box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
                box_probs = box_utils.nms(box_probs, self.nms_method,
                                        score_threshold=prob_threshold,
                                        iou_threshold=self.iou_threshold,
                                        sigma=self.sigma,
                                        top_k=top_k,
                                        candidate_size=self.candidate_size)
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.size(0))
            if not picked_box_probs:
                return torch.tensor([]), torch.tensor([]), torch.tensor([])
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, 0] *= widths[i]
            picked_box_probs[:, 1] *= heights[i]
            picked_box_probs[:, 2] *= widths[i]
            picked_box_probs[:, 3] *= heights[i]
            all_box_coord.append(picked_box_probs[:, :4])
            all_labels.append(torch.tensor(picked_labels))
            all_box_probs.append(picked_box_probs[:, 4])
        return all_box_coord, all_labels, all_box_probs



def create_mobilenetv1_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor