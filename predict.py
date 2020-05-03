import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch.optim as optim
import copy
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
from time import time
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse 

parser = argparse.ArgumentParser(description = 'Prediction Module')

parser.add_argument('--dir', type = str, default= 'flowers/test/10/image_07090.jpg',  help = 'Location of data files predicting')
parser.add_argument('--topk_val', type = int,  help = 'to print out the top K classes along with associated probabilities')
parser.add_argument('--json_file', type = str,  help = 'load a JSON file that maps the class values to other category names')
parser.add_argument('--GPU', type = str,  help = 'Using GPU')

args = parser.parse_args()
image_path = args.dir

if args.topk_val:
    topk = args.topk_val
else: 
    topk = 5


# Loading JSON file 
if args.json_file:  
    with open('args.json_file', 'r') as f:
        cat_to_name = json.load(f)
else: 
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

# selecting which device to run commands
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'


#Loading the checkpoint from the training script 
def load_checkpoint(filepath):
    
    #load checkpoint 
    checkpoint = torch.load(filepath)
    
    #load arch model and freeze paramters 
    if checkpoint ['arch']  == ' alexnet':
        models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        classifier = nn.Sequential(OrderedDict([
	                  ('fc1', nn.Linear(9216, 4096)),
	                  ('relu', nn.ReLU()),
	                  ('fc2', nn.Linear(4096, 102)),
	                  ('output', nn.LogSoftmax(dim=1))
	                  ]))
        model.classifier = classifier


    elif checkpoint ['arch']  == ' vgg11':
        models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(25088, hd)),
                      ('relu', nn.ReLU()),
                      ('fc2', nn.Linear(hd, 102)),
                      ('output', nn.LogSoftmax(dim=1))
                      ])) 

        model.classifier = classifier

    else:
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
	                  ('fc1', nn.Linear(25088, 4096)),
	                  ('relu', nn.ReLU()),
	                  ('fc2', nn.Linear(4096, 102)),
	                  ('output', nn.LogSoftmax(dim=1))
	                  ]))

        model.classifier = classifier
    
    
    model.class_to_idx = checkpoint['mapping']
    
    model.clssidifer = checkpoint['classifier']
    
    model.state_dict(checkpoint['state_dict'])  
 
    return model



#image preproessing 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #opening image
    im = Image.open(image)

    #current size of image 
    width, height = im.size
    
    #resizing images 
    (width,height) = (int(im.width//2), int(im.height/2))
    

    im_resized = im.resize((width,height))
    
    #new size of image 
    width, height = im.size 
    
    #crop out the center 224x224 portion of the image
    left = (width -224) /2
    upper = (height - 224)/2
    right = left + 224
    lower = upper + 224
    im = im.crop((left, upper, right,lower))
    
    # converting Color Channels of images
    np_image = np.array(im)/255
    
    # Normalizing image
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])

    # changing color channel to be in the first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
        
# Class Prediction
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #procinging image 
    image = process_image(image_path)
    
    # Changing numpy arry to Tensor  
    if device == 'cuda':
    	im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
    	im = torch.from_numpy (image).type (torch.FloatTensor)
        
    im = im.unsqueeze(0)
    
    model.to (device)
    im.to(device)
    
    # turning off gradient
    with torch.no_grad ():
        outputs = model.forward(im)

    prob = torch.exp(outputs)
    
    probs, label = prob.topk(topk)
    probs = probs.numpy().tolist()[0]
    label = label.numpy().tolist()[0]
    
    ind_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    classes = [ind_to_class [item] for item in label]
    classes = np.array (classes) 
    
    return probs, classes
        

model = load_checkpoint('checkpoint.pth')

probs, classes = predict(image_path, model, topk, device)

class_names = [cat_to_name [item] for item in classes]

print('The Image class is: ', class_names, '/n with a Probability of:', probs)
