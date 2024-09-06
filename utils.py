# utils.py

import torch
from torchvision import models
from collections import OrderedDict
import json
import os
from PIL import Image
import numpy as np

def load_checkpoint(filepath):
    """
    Load a checkpoint and rebuild the model.
    
    Parameters:
    - filepath (str): Path to the checkpoint file.
    
    Returns:
    - model (torch.nn.Module): The reconstructed model.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    arch = checkpoint['architecture']
    
    # Load a pre-trained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'")
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild the classifier
    if arch == 'vgg16':
        input_size = 25088
        classifier = checkpoint['classifier']
        model.classifier = classifier
    elif arch == 'resnet18':
        input_size = 512
        classifier = checkpoint['classifier']
        model.fc = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """
    Process an image for use in a PyTorch model.
    
    Parameters:
    - image_path (str): Path to the image file.
    
    Returns:
    - img_tensor (torch.Tensor): The processed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    
    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    
    image = preprocess(image)
    return image

def predict(image_path, model, topk=5, device='cpu'):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Parameters:
    - image_path (str): Path to the image file.
    - model (torch.nn.Module): The trained model.
    - topk (int): Number of top predictions to return.
    - device (str): Device to perform computations ('cpu' or 'cuda').
    
    Returns:
    - probs (list): Probabilities of the top K classes.
    - classes (list): Top K class labels.
    """
    model.to(device)
    model.eval()
    
    # Process image
    img = process_image(image_path)
    img = img.unsqueeze(0)  # Add batch dimension
    img = img.to(device)
    
    with torch.no_grad():
        output = model(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    
    top_p = top_p.cpu().numpy().flatten()
    top_class = top_class.cpu().numpy().flatten()
    
    # Invert the class_to_idx dictionary
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[c] for c in top_class]
    
    return top_p, classes
