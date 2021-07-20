from pytorchyolo import detect, models
import cv2

class yoloPredictor():

    def __init__(self,
                 cfg="/homes/15hagge/deepActiveLearning/PyTorch-YOLOv3/config/robocup.cfg",
                 weights="/homes/15hagge/deepActiveLearning/DeepActiveLearning/checkpoints/yolov3_ckpt_199.pth"):
        self.cfg = cfg
        self.weights = weights
        self.model = models.load_model(self.cfg, self.weights)
        self.conf_thresh = 0.1

    def predict(self, imgpath):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = detect.detect_image(self.model, img, conf_thres=0.1)
        return boxes  # [[x1, y1, x2, y2, confidence, class]]

    def predictFromLoadedImage(self, img):
        boxes = detect.detect_image(self.model, img, conf_thres=0.1)
        return boxes  # [[x1, y1, x2, y2, confidence, class]]

