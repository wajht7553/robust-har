# LOSO Training Framework for Human Activity Recognition

This framework provides a complete training pipeline for Human Activity Recognition (HAR) using Leave-One-Subject-Out (LOSO) cross-validation.

## Models Implemented

1. **DeepConvLSTM**: CNN-LSTM hybrid architecture ([Ordonez & Roggen, 2016](https://www.mdpi.com/1424-8220/16/1/115))
2. **MobileViT**: Mobile Vision Transformer adapted for time-series ([Mehta & Rastegari, 2021](https://arxiv.org/abs/2110.02178))

## Project Structure

```
repo/
├── dataset/
    └── processed_acc_gyr/
        ├── X.npy                 # Windowed data (N, 200, 6)
        ├── y.npy                 # Labels (N,)
        └── subject_index.json    # Subject-to-sample mapping
├── models/
│   ├── MobileViT.py              # MobileViT implementation
│   ├── DeepConvLSTM.py           # DeepConvLSTM implementation
│    __init__.py
├── utils/
│   ├── dataset_loader.py         # LOSO data loading utilities
│   ├── training_utils.py         # Training loops and metrics
│   └── __init__.py
implementation
├── train_loso.py                 # Main training script
└── preprocess.py                 # Data preprocessing

```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Data (if not already done)

```bash
python preprocess.py
```

This creates windowed data in `dataset/processed_acc_gyr/`:
- Window size: 4 seconds (200 samples at 50 Hz)
- Stride: 2 seconds (50% overlap)
- 6 channels: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
- 8 activities: walking, running, sitting, standing, lying, climbingup, climbingdown, jumping

### 2. Train Models with LOSO Validation

**Train DeepConvLSTM:**
```bash
python train_loso.py --model deepconvlstm --epochs 50 --batch_size 32 --lr 0.001
```

**Train MobileViT:**
```bash
python train_loso.py --model mobilevit --epochs 50 --batch_size 32 --lr 0.001
```

**Train Both Models:**
```bash
python train_loso.py --model both --epochs 50 --batch_size 32 --lr 0.001
```

### 3. Results

Results are saved in `results/<model_name>_<timestamp>/`:
- `results.json`: Complete results including:
  - Per-subject metrics (accuracy, F1, precision, recall, confusion matrix)
  - Training history (losses and accuracies per epoch)
  - Aggregate statistics (mean ± std across all subjects)
- `model_<subject>.pt`: Trained model checkpoints for each LOSO fold

## Command-Line Arguments

```
--model         Model to train: 'deepconvlstm', 'mobilevit', or 'both' (required)
--epochs        Number of epochs per fold (default: 50)
--batch_size    Batch size (default: 32)
--lr            Learning rate (default: 0.001)
--data_dir      Path to processed data (default: 'dataset/processed_acc_gyr')
--results_dir   Path to save results (default: 'results')
```

## Architecture Details

### DeepConvLSTM
- 4 convolutional blocks (64 filters, kernel size 11)
- Batch normalization
- 2-layer LSTM (128 units each)
- Dropout (0.5)
- Early stopping (patience: 10 epochs)
- ~500K parameters

### MobileViT
- Stem + 3 stages with inverted residuals
- 2 MobileViT blocks with transformers (4 heads, 2-4 layers)
- Global average pooling
- Dimensions: [32, 64, 96, 128]
- Early stopping (patience: 10 epochs)
- ~300K parameters

## Evaluation

### LOSO Cross-Validation
- Each subject is held out once as test set
- Remaining subjects form training set
- Normalization computed on training set, applied to test set
- Final metrics: mean ± std across all subjects

### Early Stopping
- Monitors validation accuracy during training
- Patience: 10 epochs (stops if no improvement for 10 consecutive epochs)
- Best model checkpoint saved automatically
- Restores best model weights after training completes
- Prevents overfitting and reduces training time

### Metrics Reported
- Accuracy
- F1-score (macro and weighted)
- Precision and Recall (per-class and averaged)
- Confusion matrix
## Customization
### Adding a New Model

1. Create model class in `models/your_model.py`
2. Implement `__init__(config)` and `forward(x)` methods
3. Add `number_of_parameters()` method
4. Update `create_model()` in `train_loso.py`
5. Add config function like `get_your_model_config()`

### Modifying Training

- Change optimizer: Edit `LOSOExperiment.train_subject_fold()`
- Add schedulers: Modify `Trainer.__init__()` and `Trainer.train()`
- Adjust early stopping patience: Change `early_stopping_patience` parameter in Trainer initialization
- Custom metrics: Extend `compute_metrics()` in `training_utils.py`
- Data augmentation: Update `HARDataset.__getitem__()` in `dataset_loader.py`

## Tips

- Use `--epochs 5` for quick testing
- GPU recommended but not required (falls back to CPU)
- For large datasets, increase `num_workers` in `create_dataloaders()`
- Results are saved incrementally (safe to interrupt)
- Each LOSO fold is independent (can parallelize manually)

## References

- Ordóñez, F.J.; Roggen, D. Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition. Sensors 2016, 16, 115.
- Mehta, S.; Rastegari, M. MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. arXiv 2021, arXiv:2110.02178.
