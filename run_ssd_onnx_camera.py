from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import onnx
import onnxruntime as ort
import numpy as np
import time
from vision.utils.misc import str2bool, Timer


def check_model(onnx_path: str):
    # Load the ONNX model
    model = onnx.load(onnx_path)
    # Check that the model is well formed
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

def camera_inference(model_path):
    check_model(model_path)

    inputs = 'input_0'
    ort_session = ort.InferenceSession(model_path)
    preds = ort_session.run(
        None, 
        {
            inputs: np.random.randn(1, 3, 300, 300).astype(np.float32)
        }
    )
    print(preds[0].shape)
    print(preds[0][0][0])

def test_camera_loop():
    cap = cv2.VideoCapture(0)
    captured_frames = 0
    begin_time = time.time()

    read_camera = True
    while read_camera:
        read_camera, image = cap.read()
        cv2.imshow('Detection output', image)
        if cv2.waitKey(1) == 27:
            break
        captured_frames +=1
        if captured_frames >= 30:
            ellapsed = time.time() - begin_time
            fps = captured_frames / ellapsed
            print(f"Fps: {fps}")
            captured_frames = 0
            begin_time = time.time()
    cv2.destroyAllWindows()

def run_detection_loop():
    timer = Timer()
    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200, device=DEVICE)
    
    timer.start("Load Model")
    net.load(model_path)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    cap = cv2.VideoCapture(0)
    captured_frames = 0
    begin_time = time.time()

    read_camera = True
    while read_camera:
        read_camera, image = cap.read()
        orig_image = image.copy()
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # print(box)
            orig_image = cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            orig_image = cv2.putText(orig_image, label,
                        (int(box[0]) + 20, int(box[1]) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        cv2.imshow('Detection output', orig_image)
        if cv2.waitKey(1) == 27:
            break

        captured_frames +=1
        if captured_frames >= 30:
            ellapsed = time.time() - begin_time
            fps = captured_frames / ellapsed
            print(f"Fps: {fps}")
            captured_frames = 0
            begin_time = time.time()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_ssd_camera.py <model path> <label path>')
        sys.exit(0)
    
    model_path = sys.argv[1]
    label_path = sys.argv[2]
    DEVICE = sys.argv[3]
    run_detection_loop()

