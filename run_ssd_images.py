import sys
sys.path.append('pytorch-ssd')
import logging
from torch_batch_predictor import create_mobilenetv1_ssd_predictor

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd #, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2

# import onnx
# import onnxruntime as ort
import numpy as np
import time
from vision.utils.misc import Timer


def run_detection_loop():
    timer = Timer()
    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=300, device=DEVICE)
    
    timer.start("Load Model")
    net.load(model_path)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    files = [f.strip() for f in open("files.list", 'r').readlines()]
    print(f"Len files: {len(files)}")

    logging.info("Main loop begin")
    timer.start("Processing")
    for i in range(0, len(files), int(BATCH_SIZE)):
        sfiles = files[i:i+int(BATCH_SIZE)]
        images = [cv2.imread(file, cv2.IMREAD_COLOR) for file in sfiles]
        

        all_boxes, all_labels, all_probs = predictor.predict_batch(images, 10, 0.4)

        if DRAW:
            for image, boxes, labels, probs in zip(images, all_boxes, all_labels, all_probs):
                orig_image = image.copy()
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
            
    ellapsed = timer.end("Processing")
    logging.info(f"Total time: {ellapsed}, fps: {ellapsed/len(files)}")
    logging.info("Done")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_ssd_camera.py <model path> <label path>')
        sys.exit(0)
    
    logging.basicConfig(level=logging.INFO)
    model_path = sys.argv[1]
    label_path = sys.argv[2]
    DEVICE = sys.argv[3]
    BATCH_SIZE = sys.argv[4]
    DRAW = True
    run_detection_loop()

