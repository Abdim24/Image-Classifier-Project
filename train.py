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

parser = argparse.ArgumentParser(description = 'Training Module')

parser.add_argument('--dir', type = str, default = 'flowers', help = 'Location of data files for Training, Validation, and Testing')
parser.add_argument('--save_check', type = str,  help = 'Location to Checkpoint file')
parser.add_argument('--arch',type=str, default='vgg16', help= 'Select model architecture to use:'
																 'Deafult = vgg16'
																 'Option 1: alexnet'
																 'Option 2: vgg11')

parser.add_argument('--hd', type= int, default= 4096, help = 'Number of Hidden Units' )

parser.add_argument('--lrt', type= float, default= 0.001, help = 'Learning rate' )
parser.add_argument('--ep',type = int, default=7, help = 'Number of epoches')
parser.add_argument('--GPU', type = str, default = 'GPU',  help = 'Using GPU')


args = parser.parse_args()

# Location of datasets(i.e. Images)
data_dir = args.dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# selecting which device to run commands
if args.GPU == 'GPU':
	device = 'cuda'
else:
	device = 'cpu'

if data_dir: 
	# transforms for training, validation, and testing data sets
	training_data_transforms = transforms.Compose([
	    transforms.RandomHorizontalFlip(),
	    transforms.RandomVerticalFlip(),  
	    transforms.RandomResizedCrop(224), 
	    transforms.ToTensor(),
	    transforms.Normalize(mean = [0.485, 0.456, 0.406],
	                         std = [0.229, 0.224, 0.225]),
	])

	validation_data_transforms = transforms.Compose([
	    transforms.RandomResizedCrop(224),
	    transforms.ToTensor(),
	    transforms.Normalize(mean = [0.485, 0.456, 0.406],
	                         std = [0.229, 0.224, 0.225]),    
	])

	testing_data_transforms = transforms.Compose([
	    transforms.RandomResizedCrop(224),
	    transforms.ToTensor(),
	    transforms.Normalize(mean = [0.485, 0.456, 0.406],
	                         std = [0.229, 0.224, 0.225]),
	])

	# Loading  datasets using ImageFolder
	training_image_datasets =  datasets.ImageFolder(train_dir,
	                                                transform = training_data_transforms)

	validation_image_datasets =  datasets.ImageFolder(valid_dir,
	                                                transform = validation_data_transforms)

	test_image_datasets =  datasets.ImageFolder(test_dir,
	                                                transform = testing_data_transforms)

	# Defining the dataloaders using the image datasets and the trainforms 
	training_dataloaders = torch.utils.data.DataLoader(training_image_datasets,
	                                                   batch_size=32,
	                                                   shuffle=True)
	validation_dataloaders = torch.utils.data.DataLoader(validation_image_datasets,
	                                                     batch_size=32,
	                                                     shuffle=True)
	test_dataloaders = torch.utils.data.DataLoader(test_image_datasets,
	                                               batch_size=32,
	                                               shuffle=True)

# Label Mapping  
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def model_select (arch, hd):
	
	if arch == ' alexnet':

		images, labels = next(iter(training_dataloaders))

		models.alexnet(pretrained=True)
		model

		for param in model.parameters():
		    param.requires_grad = False 

		if hd:
		    
			classifier = nn.Sequential(OrderedDict([
			                          ('fc1', nn.Linear(9216, hd)),
			                          ('relu', nn.ReLU()),
			                          ('fc2', nn.Linear(hd, 102)),
			                          ('output', nn.LogSoftmax(dim=1))
			                          ]))
		else: 

			classifier = nn.Sequential(OrderedDict([
			                          ('fc1', nn.Linear(9216, 4096)),
			                          ('relu', nn.ReLU()),
			                          ('fc2', nn.Linear(4096, 102)),
			                          ('output', nn.LogSoftmax(dim=1))
			                          ]))

		model.classifier = classifier
		model

	elif arch == 'vgg11':

		images, labels = next(iter(training_dataloaders))

		models.vgg11(pretrained=True)
		model

		for param in model.parameters():
		    param.requires_grad = False 

		if hd:
		    
			classifier = nn.Sequential(OrderedDict([
			                          ('fc1', nn.Linear(25088, hd)),
			                          ('relu', nn.ReLU()),
			                          ('fc2', nn.Linear(hd, 102)),
			                          ('output', nn.LogSoftmax(dim=1))
			                          ]))
		else: 

			classifier = nn.Sequential(OrderedDict([
			                          ('fc1', nn.Linear(25088, 4096)),
			                          ('relu', nn.ReLU()),
			                          ('fc2', nn.Linear(4096, 102)),
			                          ('output', nn.LogSoftmax(dim=1))
			                          ]))

		model.classifier = classifier
		model

	else: 

		images, labels = next(iter(training_dataloaders))

		model = models.vgg16(pretrained=True)
		model

		for param in model.parameters():
		    param.requires_grad = False 

		if hd: 
		    
			classifier = nn.Sequential(OrderedDict([
			                          ('fc1', nn.Linear(25088, hd)),
			                          ('relu', nn.ReLU()),
			                          ('fc2', nn.Linear(hd, 102)),
			                          ('output', nn.LogSoftmax(dim=1))
			                          ]))
		else:
			classifier = nn.Sequential(OrderedDict([
			                          ('fc1', nn.Linear(25088, 4096)),
			                          ('relu', nn.ReLU()),
			                          ('fc2', nn.Linear(4096, 102)),
			                          ('output', nn.LogSoftmax(dim=1))
			                          ]))

		model.classifier = classifier
		model

	return model


model = model_select (args.arch, args.hd)
# initializing loss function to criterion variable  
criterion = nn.NLLLoss()

# Selecting Learing rate(lr) and optimizing lr step size 
if args.lrt:
	optimizer = optim.SGD(model.classifier.parameters(), lr=args.lrt, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
else: 
	optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# selecting number of epochs 
if args.ep: 
	epochs = args.ep
else:
	epochs = 7 

# traing model function 
def train_model(model, criterion, optimizer,scheduler, epochs):
    
    mode =  ""
    count = 0
    epoch_total = epochs

    for epoch in range(epochs):
        count += 1
        print("\nepoch:", count, "/", epoch_total)
        
        Training_loss = 0.0
        Training_Accuracy = 0
        
        Validation_loss = 0.0
        Validation_Accuracy = 0
        
        model.train()

        # Iterate over data.
        for ii, (images,labels) in  enumerate (training_dataloaders):

            images = images.to(device)
            labels = labels.to(device)

           # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


            # statistics
            Training_loss += loss.item() 
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            Training_Accuracy += equality.type(torch.FloatTensor).mean()

            epoch_loss_Training = Training_loss/len(training_dataloaders)
            epoch_acc_Training = Training_Accuracy.double()/len(training_dataloaders)*100
                
        model.eval()

        with torch.no_grad():
            # Iterate over data.
            for ii, (images,labels) in enumerate (validation_dataloaders):

                images = images.to(device)
                labels = labels.to(device)

                # forward
                outputs = model.forward(images)
                loss = criterion(outputs, labels)

                # statistics
                Validation_loss += loss.item()  
                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim=1)[1])
                Validation_Accuracy += equality.type(torch.FloatTensor).mean()

                epoch_loss_Validation = Validation_loss / len(validation_dataloaders)
                epoch_acc_Validation = Validation_Accuracy.double() / len(validation_dataloaders)*100
                    
        scheduler.step()
                    
        print("Taining loss", epoch_loss_Training," ", "Accuracy", epoch_acc_Training )
        print("Validation", epoch_loss_Validation," ", "Accuracy", epoch_acc_Validation )
             
    return model

# training the model 
model.to (device)
model = train_model(model, criterion, optimizer, scheduler, epochs)

# testing trained model Accuracy 
Test_loss = 0.0
Test_Accuracy = 0

model.eval()
with torch.no_grad():

    for ii, (images,labels) in enumerate (test_dataloaders):

        images = images.to('cuda')
        labels = labels.to('cuda')

        # forward
        outputs = model.forward(images)

        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        Test_Accuracy += equality.type(torch.FloatTensor).mean()


        epoch_acc_test = Test_Accuracy.double() / len(test_dataloaders)*100
print('The Model has an Accuracy of ', epoch_acc_test, ' ', '%' )

model.state_dict()

# Saving Model Checkpoint 
model.class_to_idx = training_image_datasets.class_to_idx
model.cpu()

checkpoint = {'arch': 'vgg16',
              'state_dict': model.state_dict(),
              'mapping':model.class_to_idx, 
              'classifier': model.classifier,}

if args.save_check:
    torch.save(checkpoint, args.save_check + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')
