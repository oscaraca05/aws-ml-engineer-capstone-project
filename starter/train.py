#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os

import argparse
import json
import logging
import sys
import io

import boto3

# PIL throws this error, if the loaded image file is truncated.
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, models, transforms

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, loss_criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            # test_loss += loss_criterion(output, target, size_average=False).item()  # sum up batch loss
            test_loss += loss_criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, loss_criterion, optimizer):
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    return model
    
def net():
    num_classes = 5
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    # loss_criterion = nn.NLLLoss()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    input_size = 224
    
    # data_dir = "s3://sagemaker-us-east-1-155431344840/deep-learning-models-on-sagemaker-project/dogImages"
    
    logger.info("Get train data loader")
    training_dir = os.path.join(args.data_dir, "train/")
    train_transform = transforms.Compose([transforms.RandomResizedCrop(input_size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(training_dir, transform = train_transform)
    subset_ix = list(range(0, 200))
    # train_data = torch.utils.data.Subset(train_data, subset_ix) # for accelerating the process
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= args.batch_size, shuffle=True)
    
    logger.info("Get test data loader")
    test_dir = os.path.join(args.data_dir , "test/")
    test_transform = transforms.Compose([transforms.RandomResizedCrop(input_size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    # test_data = torch.utils.data.Subset(test_data, subset_ix) # for accelerating the process
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= args.test_batch_size)
    
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
