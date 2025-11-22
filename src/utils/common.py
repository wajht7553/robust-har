import torch
import json
import os


def save_model(model, filepath):
    """Save model state dict"""
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath, device):
    """Load model state dict"""
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model


def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)
