
import os
import argparse
import torch
import torch.nn as nn
import copy
from src.models.factory import create_model
from src.data.splitter import create_dataloaders
from src.utils.metrics import compute_metrics
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def fuse_modules(model):
    # Fuse Conv+BN+ReLU (or SiLU if supported, but SiLU fusion is tricky in older PT)
    # For now, we'll skip explicit fusion if not using standard layers or just rely on dynamic quantization
    # But for static, we usually fuse.
    # MobileViT uses ConvBNActivation which has Conv+BN+SiLU.
    # We can try to fuse Conv+BN.
    
    for m in model.modules():
        if type(m).__name__ == 'ConvBNActivation':
            # Access the internal sequential block
            # self.conv = nn.Conv1d...
            # self.bn = nn.BatchNorm1d...
            # self.activation = ...
            # Since they are attributes, not a Sequential, we can't use torch.quantization.fuse_modules directly on 'm'
            # unless we restructure.
            # However, 'm' has a forward method that calls them.
            pass
            
    # Standard fusion for Sequential modules if any
    # torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained float model .pt file")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--data_dir", type=str, default="dataset/processed_acc_gyr", help="Path to dataset")
    parser.add_argument("--quant_mode", type=str, default="dynamic", choices=["dynamic", "static"], help="Quantization mode")
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    
    # Create Model
    model = create_model("mobilevit", config)
    
    # Load Weights
    # The saved model might be the full model or state_dict. 
    # Based on train.py: save_model(model, ...) uses torch.save(model, ...) or state_dict?
    # Let's check train.py. It imports save_model from src.utils.common.
    # Assuming it saves the whole model or we need to load state_dict.
    # Let's try loading state_dict first, if fails, load full model.
    
    try:
        state_dict = torch.load(args.model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict # It was the full model
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    print("Original Model Size:")
    print_size_of_model(model)

    # Prepare Data for Calibration (needed for static) or Evaluation
    # We need a small calibration set.
    # Let's load a small amount of data.
    # We can use the LOSOSplitter but just take one fold or a subset.
    from src.data.splitter import LOSOSplitter
    splitter = LOSOSplitter(args.data_dir)
    # Just take the first fold
    splits = list(splitter.get_loso_splits())
    if not splits:
        print("No data found.")
        return
        
    subject, X_train, y_train, X_test, y_test = splits[0]
    
    # Create dataloader
    _, test_loader, _ = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32)
    
    if args.quant_mode == "dynamic":
        print("Performing Dynamic Quantization...")
        # Quantize Linear and LSTM/RNN layers. Conv layers are not quantized in dynamic quantization usually.
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
        )
        
    elif args.quant_mode == "static":
        print("Performing Static Quantization...")
        # Static quantization requires model modification (QuantStub/DeQuantStub) and fusion.
        # Since we haven't modified the model definition to include Stubs, 
        # we can try to wrap it or use a workflow that inserts them, but that's complex.
        # For now, let's stick to Dynamic Quantization as a first step for "TinyML" on CPU/Edge.
        # Or we can use QConfig and prepare/convert.
        
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate
        print("Calibrating...")
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if i > 10: break # Calibrate on a few batches
                model(inputs)
        
        torch.quantization.convert(model, inplace=True)
        quantized_model = model

    print("Quantized Model Size:")
    print_size_of_model(quantized_model)
    
    # Evaluate
    print("Evaluating Quantized Model...")
    # We need to ensure the model is on CPU for quantized inference usually
    quantized_model.to('cpu')
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to('cpu')
            outputs = quantized_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")
    
    # Save Quantized Model
    save_path = args.model_path.replace(".pt", f"_quant_{args.quant_mode}.pt")
    # TorchScript is preferred for quantized models
    try:
        scripted_model = torch.jit.script(quantized_model)
        torch.jit.save(scripted_model, save_path)
        print(f"Saved quantized model to {save_path}")
    except:
        torch.save(quantized_model, save_path)
        print(f"Saved quantized model (not scripted) to {save_path}")

if __name__ == "__main__":
    main()
