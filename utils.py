from config import LABELS
from PIL import Image
from xml.etree import ElementTree as ET


def get_img_data(xml_path: str, data_path: str):
    """
    The function extracts data from XML files.
    Arguments:
        xml_path (str) - Path to a XML file.
        data_path (str) - Path to a specific data directory.
    Returns:
        img (PIL.Image) - A loaded image based on path from XML file.
        labels  (list) - List with labels of bounding boxes.
        boxes (list) - List with bounding boxes coordinates in Pascal_VOC format.
    """

    root = ET.parse(xml_path).getroot()

    img_filename = root.find("filename").text

    img = Image.open(data_path + img_filename)
    labels = []
    boxes = []

    objects = root.findall("object")

    for object_ in objects:
        label = object_.find("name").text
        label = LABELS[label]

        bnd_box = object_.find("bndbox")
        xmin = int(bnd_box.find("xmin").text)
        ymin = int(bnd_box.find("ymin").text)
        xmax = int(bnd_box.find("xmax").text)
        ymax = int(bnd_box.find("ymax").text)

        labels.append(label)
        boxes.append([xmin, ymin, xmax, ymax])

    return img, labels, boxes