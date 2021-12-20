import numpy as np
import torch
import cv2


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ResizeBatch(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, images, boxes=None, labels=None):
        results = []
        for image in images:
            results.append(cv2.resize(image, (self.size,
                                 self.size)))
        return np.array(results, dtype=np.uint8), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(0, 3, 1, 2), boxes, labels