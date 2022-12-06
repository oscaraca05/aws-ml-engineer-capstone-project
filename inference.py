import logging
import os
import sys
import io


# PIL throws this error, if the loaded image file is truncated.
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    num_classes = 5
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def model_fn(model_dir):
    # logger.info("model_fn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = net()
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    logger.info("loading model abc...")
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    logger.info("model abc loaded!")
    
    return model.to(device)  

def input_fn(request_body, content_type):
    image = Image.open(io.BytesIO(request_body))

    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    return transformation(image).unsqueeze(0)