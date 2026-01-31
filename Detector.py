from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self,model_version="v8",model_variant="m",conf=0.25,iou=0.45,device=None):
        self.model_path = f"yolo{model_version}{model_variant}.pt"
        self.model = YOLO(self.model_path)
        self.conf = conf
        self.iou = iou

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
