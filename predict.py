from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import torchvision.transforms as T
from model import get_model
from config import DEVICE, LABELS, NUM_LABELS, IMG_SIZE
from PIL import Image
from torchvision.ops import nms


def parse_args():
    """
    The function parses arguments passes by user in run query.

    Returns:
        args (Namespace) - Arguments.
    """
    parser = ArgumentParser(description="Use script for detect objects on image.")

    parser.add_argument("--img_path", help="Path to the image.", type=str)
    parser.add_argument(
        "--model-path",
        help="Path to the specific model checkpoint",
        default="./outputs/model.pth",
        type=str,
    )
    args = parser.parse_args()

    return args


def predict(model, path: str):
    """
    The function predicts bounding boxes for passed image.

    Arguments:
        model (nn.Module) - The model.
        path (str) - Path to the image.
    """
    model.eval()

    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    img = Image.open(path)
    img_sh = np.array([img.width, img.height, img.width, img.height])

    img_t = transform(img)
    img_t = [img_t.to(DEVICE)]
    img_t_sh = np.array([IMG_SIZE, IMG_SIZE, IMG_SIZE, IMG_SIZE])

    predictions = model(img_t)

    boxes = predictions[0]["boxes"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]

    # Apply NMS on predictions
    idx = nms(boxes, scores, 0.05)

    # Get appropriate predictions based on NMS outputs
    boxes = boxes[idx].cpu().detach().numpy()
    labels = labels[idx].cpu().detach().numpy()
    scores = scores[idx].cpu().detach().numpy()

    # Resize boxes to original image
    boxes = (boxes / img_t_sh) * img_sh

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(img)

    # Add rectangles to image
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        score = round(float(scores[i]) * 100, 2)

        label_text = str(list(LABELS.keys())[label])
        bbox_text = label_text + ": " + str(score) + "%"

        box_width = box[2] - box[0]
        box_height = box[3] - box[1]

        rect = patches.Rectangle(
            (box[0], box[1]),
            box_width,
            box_height,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1], bbox_text)

    plt.show()
    fig.savefig("outputs/predictions.png")


if __name__ == "__main__":
    args = parse_args()

    img_path = args.img_path
    model_path = args.model_path

    assert (img_path is not None) and (type(img_path) is str), "Image path not defined!"

    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    predict(model, img_path)
