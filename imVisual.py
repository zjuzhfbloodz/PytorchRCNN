import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, make_grid
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt



def readImage(path, size):
    mode = Image.open(path)
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    mode = transform(mode)
    return mode


def readImageTensor(path, size):
    mode = read_image(path)
    transform = torch.nn.Sequential(
        T.Resize(size))
    mode = transform(mode)
    return mode


def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
 
    return (b,g,r)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(50,50))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def plot_image(image, boxes, labels, lb_names, lb_colors, lb_infos=None, save_name='', width=2, font_size=20):
    """
    Draws bounding boxes on given image.
    Args:
    image (Image): `Tensor`, `PIL Image` or `numpy.ndarray`.
    boxes (Optional[Tensor]): `FloatTensor[N, 4]`, the boxes in `[x1, y1, x2, y2]` format.
    labels (Optional[Tensor]): `Int64Tensor[N]`, the class label index for each box.
    lb_names (Optional[List[str]]): All class label names.
    lb_colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of all class label names.
    lb_infos (Optional[List[str]]): Infos for given labels.
    save_name (Optional[str]): Save image name.
    """
    if not isinstance(image, torch.Tensor):
        image = torchvision.transforms.ToTensor()(image)

    if boxes is not None:
        if image.dtype != torch.uint8:
            image = torchvision.transforms.ConvertImageDtype(torch.uint8)(image)
        draw_labels = None
        draw_colors = None
        if labels is not None:
            draw_labels = [lb_names[str(int(i))] for i in labels] if lb_names is not None else None
            draw_colors = [lb_colors[str(int(i))] for i in labels] if lb_colors is not None else None
        if draw_labels and lb_infos:
              draw_labels = [f'{l} {i}' for l, i in zip(draw_labels, lb_infos)]
        # draw boxes
        res = draw_bounding_boxes(image, boxes,
          labels=draw_labels, colors=draw_colors, width=width, font_size=font_size)
    else:
        res = image
        
    show([res])
   
    if save_name:
        res = res.permute(1, 2, 0).contiguous().numpy()
        Image.fromarray(res).save(save_name)


# define names and colors for coco dataset
names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}
colors = dict([(i,random_color()) for i in names.keys()])