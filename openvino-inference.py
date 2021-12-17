import cv2
import logging
import time
import numpy as np
import sys

from imutils.video import VideoStream
from imutils.video import FPS

from openvino_utils import (
    import_network,
    request_sync
)
from openvino_predictor import Predictor


class NetWrapper:
    def __init__(self, exec_net):
        self.net = exec_net

    def to(self, *args, **kwargs):
        pass

    def eval(self):
        pass

    def forward(self, images):
        preds = request_sync(self.net, images)
        return preds['scores'], preds['boxes']


def main():
    logging.info("Loading executable network")
    exec_net = import_network(
        model_path,
        num_requests=1
    )
    net = NetWrapper(exec_net)
    predictor = Predictor(
        net=net, 
        size=300,
        candidate_size=200,
        nms_method=None,
        mean=np.array([127, 127, 127]),
        std=128.0,
        iou_threshold = 0.45,
        sigma=.5)
    class_names = [name.strip() for name in open(label_path).readlines()]
    logging.info("Initializing video stream")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    logging.info("Main loop begin")
    while True:
        frame = vs.read()
        orig_image = frame.copy()

        boxes, labels, probs = predictor.predict(frame, 10, 0.4)

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(orig_image, label,
                        (int(box[0]) + 20, int(box[1]) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

        cv2.imshow("Frame", orig_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
             break
        fps.update()

    fps.stop()
    logging.info("elapsed time: {:.2f}".format(fps.elapsed()))
    logging.info("approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()
    logging.info("Done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    model_path = sys.argv[1]
    label_path = sys.argv[2]
    main()