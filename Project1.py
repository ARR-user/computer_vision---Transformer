import numpy as np



#import torch which has many of the functions to build deep learning models and to train them
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import torchvision, which was lots of functions for loading and working with image data
import torchvision
import torchvision.transforms as transforms

import tqdm

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384,256 )
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64 )
        self.fc4 = nn.Linear(64,10)

        self.relu=nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        y = self.fc4(x)
        
       
        return y



class MyClassifier():
    
    ''' Do not change the class name. Do not change any of the existing function names. You may add extra functions as you see fit.'''
    
    def __init__(self):
        self.class_labels = ['edible_1', 'edible_2', 'edible_3', 'edible_4', 'edible_5',
                            'poisonous_1', 'poisonous_2', 'poisonous_3', 'poisonous_4', 'poisonous_5']
        
        
    def setup(self):

        image_means=[0.42120951861354433,0.39622252713640727,0.32426305079647155]
        image_stds=[0.258645285220707,0.243852862060128,0.24828967822747286]
                
        ''' This function will initialise your model. 
            You will need to load the model architecture and load any saved weights file your model relies on.
        '''


        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224, 224)), 
         transforms.Normalize(image_means, image_stds)])
    
        train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.05, contrast=0.05,saturation=0,hue=0),  # Subtle color variations check if
    
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.005),  # Add manual  Gaussian noise
       
     
    ])



        
        self.dino= torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
        self.dino.eval()

        
        self.model=DeepNet()
        self.model.load_state_dict(torch.load('test.pth'))
        self.model.eval()
        pass
        
    def test_image(self, image):
        ''' This function will be given a PIL image, and should return the predicted class label for that image. 
            Currently the function is returning a random label.
                
        '''
        im_tensor=self.transform(image).unsqueeze(0)
        feature=self.dino(im_tensor)
        prediction=self.model(feature)
        predicted_idx=torch.argmax(prediction)
        predicted_cls = self.class_labels[predicted_idx]
        
        return predicted_cls
        
