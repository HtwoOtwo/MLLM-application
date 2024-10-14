import json

import torch
import yaml
from ultralytics import YOLO

with open("function.yaml", "rb") as function_file:
    functionconfig = yaml.safe_load(function_file)
labels_spec = functionconfig["metadata"]["annotations"]["spec"]
LABELS_DICT = {item["id"]: item["name"] for item in json.loads(labels_spec)}

class DetectionModel:
    def __init__(self, args):
        self.args = args
        self.model = self.load_model(args)

    def load_model(self, args):
        raise NotImplementedError

    def parse_result(self, result):
        raise NotImplementedError

    def __call__(self, image_path, prompt):
        return self.parse_result(self.model(image_path, prompt))


class Yolov8DetectionModel(DetectionModel):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self, args):
        model = YOLO(args.model_path)
        return model

    def __call__(self, image_path, *args):
        return self.parse_result(self.model(image_path, device=self.args.device, verbose=False))

    def parse_result(self, result):
        result = result[0]
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            if int(cls) in LABELS_DICT.keys() and conf > 0.35: # TODO
                return {"answer_yes": True, "output": result.path}
        return {"answer_yes": False, "output": result.path}


def init_yolov8_model():
    # Define default values
    model_path = "model.pt"
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Create a simple class to hold the arguments
    class Args:
        def __init__(self, model_path, device):
            self.model_path = model_path
            self.device = device

    # Return an instance of Args with the default values
    return Args(model_path, device)
