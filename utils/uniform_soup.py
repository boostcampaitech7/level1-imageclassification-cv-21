import inspect
import os
import sys
import time
import numpy as np

def uniform_soup(model, path, device = "cpu", by_name = False):
    try:
        import torch
    except:
        print("If you want to use 'Model Soup for Torch', please install 'torch'")
        return model
        
    if not isinstance(path, list):
        path = [path]
    model = model.to(device)
    model_dict = model.state_dict()
    soups = {key:[] for key in model_dict}
    for i, model_path in enumerate(path):
        weight = torch.load(model_path, map_location = device)
        weight_dict = weight.state_dict() if hasattr(weight, "state_dict") else weight
        if by_name:
            weight_dict = {k:v for k, v in weight_dict.items() if k in model_dict}
        for k, v in weight_dict.items():
            soups[k].append(v)
    if 0 < len(soups):
        soups = {k:(torch.sum(torch.stack(v), axis = 0) / len(v)).type(v[0].dtype) for k, v in soups.items() if len(v) != 0}
        model_dict.update(soups)
        model.load_state_dict(model_dict)
    return model