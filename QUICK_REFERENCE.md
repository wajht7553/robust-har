# Quick Reference Card - LOSO Training Framework

## Installation
```bash
pip install -r requirements_training.txt
```

## Basic Commands

### Train DeepConvLSTM
```bash
python train_loso.py --model deepconvlstm --epochs 50 --batch_size 32 --lr 0.001
```

### Train MobileViT
```bash
python train_loso.py --model mobilevit --epochs 50 --batch_size 32 --lr 0.001
```

### Train Both Models
```bash
python train_loso.py --model both --epochs 50 --batch_size 32
```

### Quick Test (Fast)
```bash
python train_loso.py --model mobilevit --epochs 5 --batch_size 64
```

### Using Batch Script (Windows)
```bash
run_training.bat mobilevit 50 32
```

## Visualize Results
```bash
# After training completes
python visualize_results.py results/mobilevit_20241116_143022/results.json

# Just print summary (no plots)
python visualize_results.py results/mobilevit_*/results.json --no_plot
```

## File Locations

| What | Where |
|------|-------|
| Training script | `train_loso.py` |
| Visualization script | `visualize_results.py` |
| Example notebook | `training_example.ipynb` |
| Models | `models/` |
| Utilities | `utils/` |
| Data | `dataset/processed_acc_gyr/` |
| Results | `results/` |
| Documentation | `TRAINING_README.md` |

## Model Configurations

### DeepConvLSTM (Default)
```python
{
    'nb_filters': 64,
    'filter_width': 11,
    'nb_units_lstm': 128,
    'nb_layers_lstm': 2,
    'nb_conv_blocks': 4,
    'drop_prob': 0.5,
    'batch_norm': True
}
```

### MobileViT (Default)
```python
{
    'dims': [32, 64, 96, 128],
    'num_transformer_layers': [2, 4],
    'patch_size': 2,
    'num_heads': 4,
    'dropout': 0.1
}
```

## Typical Results

| Metric | DeepConvLSTM | MobileViT |
|--------|--------------|-----------|
| Accuracy (LOSO) | 80-85% | 82-87% |
| Parameters | ~500K | ~300K |
| Time/fold | 5-10 min | 8-15 min |
| Total time (15 folds) | ~1.5 hours | ~2.5 hours |

## Common Issues

### Issue: ModuleNotFoundError
**Solution**: Run from repo root, not subdirectories
```bash
cd c:\Users\DIPLAB\Desktop\Work\Shehzad\IEEETCE\repo
python train_loso.py --model mobilevit
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size
```bash
python train_loso.py --model mobilevit --batch_size 16
```

### Issue: Data not found
**Solution**: Run preprocessing first
```bash
python preprocess.py
```

### Issue: Training too slow
**Solution**: Use GPU or reduce epochs for testing
```bash
python train_loso.py --model mobilevit --epochs 10
```

## Output Structure

```
results/
└── mobilevit_20241116_143022/
    ├── results.json                    # All metrics
    ├── model_proband1.pt               # Trained model (fold 1)
    ├── model_proband2.pt               # Trained model (fold 2)
    ├── ...
    └── mobilevit_loso_results.png      # Visualization
```

## Python API (Programmatic Use)

```python
from train_loso import LOSOExperiment, get_mobilevit_config

# Create experiment
config = get_mobilevit_config(window_size=200, nb_channels=6, nb_classes=8)
experiment = LOSOExperiment('mobilevit', config, results_dir='my_results')

# Run LOSO
results = experiment.run_loso(epochs=50, batch_size=32, lr=1e-3)

# Access results
print(f"Mean accuracy: {results['aggregate_metrics']['mean_accuracy']:.4f}")
```

## Customization Examples

### Change learning rate
```bash
python train_loso.py --model mobilevit --lr 0.0005
```

### Change number of epochs
```bash
python train_loso.py --model mobilevit --epochs 100
```

### Change batch size
```bash
python train_loso.py --model mobilevit --batch_size 64
```

### Adjust early stopping patience (requires code edit)
Edit `train_loso.py`, line with `Trainer(...)`:
```python
trainer = Trainer(
    model, self.device, criterion, optimizer, 
    early_stopping_patience=15,  # Change from 10 to 15
    checkpoint_path=checkpoint_path
)
```

### Use different data directory
```bash
python train_loso.py --model mobilevit --data_dir path/to/data
```

### Save results elsewhere
```bash
python train_loso.py --model mobilevit --results_dir my_experiments
```

## Monitoring Progress

During training you'll see:
```
Epoch 25 [Train]: 100%|████████| 576/576 [00:45<00:00, loss=0.2134, avg_loss=0.2145]
Epoch 25 [Val]:   100%|████████| 32/32 [00:02<00:00, loss=0.4521, avg_loss=0.4512]
Epoch 25/50 - Train Loss: 0.2145, Train Acc: 0.9234 | Val Loss: 0.4512, Val Acc: 0.8512
```

## Quick Checks

### Verify data is ready
```python
import numpy as np
X = np.load('dataset/processed_acc_gyr/X.npy')
print(f"Data shape: {X.shape}")  # Should be (N, 200, 6)
```

### Check GPU availability
```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU available
```

### Test model creation
```python
from models.MobileViT import MobileViT
config = {'window_size': 200, 'nb_channels': 6, 'nb_classes': 8,
          'dims': [32, 64, 96, 128], 'num_transformer_layers': [2, 4],
          'patch_size': 2, 'num_heads': 4, 'dropout': 0.1}
model = MobileViT(config)
print(f"Parameters: {model.number_of_parameters():,}")
```

## Help

For detailed documentation:
- Read `TRAINING_README.md`
- Check `IMPLEMENTATION_SUMMARY.md`
- Try `training_example.ipynb`

For command-line help:
```bash
python train_loso.py --help
python visualize_results.py --help
```

---
**Need help?** Check the documentation or review the example notebook.
