from pytorchyolo import detect, models
import cv2
import yaml

class yoloPredictor():

    def __init__(self):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.cfg = config['cfg_path']
        self.weights = config['last_weights']
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

