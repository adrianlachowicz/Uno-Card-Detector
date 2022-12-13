import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import NUM_LABELS


def get_model():
    """
    The function prepares FasterRCNN model.

    Returns:
        model (nn.Module) - The Faster RCNN model.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_LABELS)
    return model
