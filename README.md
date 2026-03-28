# RobustHAR

A comprehensive framework for **Robust Human Activity Recognition (HAR)** using Deep Learning. This repository utilizes a structured, modular design to experiment with various models, datasets, and degradation/robustness strategies using Leave-One-Subject-Out (LOSO) Cross-Validation.

## Features
- **Modular Configuration**: Managed by [Hydra](https://hydra.cc/), allowing easy combination of models, datasets, and training strategies.
- **Robust Evaluation**: Evaluates HAR models not only on clean data but also under strategies like signal degradation or modality dropout.
- **Robust LOSO Cross-Validation**: Holds out varying subjects for testing and validation to measure true generalization.
- **Modern Architectures Support**: Includes MobileViT, Mamba, DeepConvLSTM, DeepConvContext, TinyHAR, and MobileNetV3.

## Environment Setup
Make sure you have PyTorch installed along with other necessary dependencies:
```bash
pip install torch numpy omegaconf hydra-core
```
*(Add other dependencies like pandas, scikit-learn as needed by your specific environment).*

## Quick Start
To train a model (e.g., MobileViT) on a dataset (e.g., PAMAP2) under a specific strategy (e.g., clean), simply run:

```bash
python train.py model=mobilevit dataset=pamap2_dataset strategy=clean
```

You can override any parameter on the command line:
```bash
python train.py model=mamba dataset=wisdm_dataset epochs=100 batch_size=64
```

---

## Codebase Overview
- `conf/`: Hydra configuration files.
  - `model/`: Configurations for individual models.
  - `dataset/`: Configurations defining dataset paths and channels.
  - `strategy/`: Testing strategies (e.g., clean, modality dropout, signal degradation).
  - `config.yaml`: The entry-point configuration file.
- `src/`: Source code.
  - `models/`: PyTorch model implementations and a `factory.py` for instantiation.
  - `data/`: Dataset wrappers, data augmentation transforms, and the robust LOSO splitter.
  - `training/`: Training loops, optimization, and trainer class.
  - `experiments/`: Experiment managers, evaluators, and the core `loso.py` pipeline.
- `train.py`: Main entry pipeline for execution.
- `dataset/`: Expected directory for data files.
- `results/`: Output directory where models and logs will be saved.

---

## 🚀 How to Add a New Dataset

1. **Preprocess your Data**
   Format your dataset to produce the following three files:
   - `X.npy`: A NumPy array of shape `(N, window_size, channels)` containing the sensor windows.
   - `y.npy`: A NumPy array of shape `(N,)` containing the class labels.
   - `subject_index.json`: A dictionary mapping each subject identifier to their respective `[start_index, end_index]` in the numpy arrays.

2. **Place the Data**
   Save these files in the dataset directory, e.g., `dataset/MyNewDataset/processed_acc_gyr/`.

3. **Create a Configuration File**
   Add a new YAML file to `conf/dataset/`, e.g., `mynewdataset.yaml`:
   ```yaml
   # @package _global_
   # Dataset Configuration
   data_dir: "dataset/MyNewDataset/processed_acc_gyr"
   input_channels: [0, 1, 2, 3, 4, 5] # E.g., Accel and Gyro 3D axes
   num_workers: 4
   classes: ["Walk", "Run", "Sit", "Stand"] # Map to your dataset's specific unique classes
   ```

4. **Run**
   ```bash
   python train.py dataset=mynewdataset model=... 
   ```

---

## 🧠 How to Add a New Model

1. **Create the PyTorch Module**
   Create a new file in `src/models/` implementing your module (e.g., `src/models/MyAwesomeModel.py`).

2. **Register in Factory**
   Open `src/models/factory.py`.
   - Import your model: `from .MyAwesomeModel import MyAwesomeModel`
   - Add it to the `create_model` function:
     ```python
     elif name == "myawesomemodel":
         return MyAwesomeModel(config)
     ```

3. **Create a Configuration File**
   Add a new YAML file to `conf/model/`, e.g., `myawesomemodel.yaml`. Provide the specific hyperparameters your model expects:
   ```yaml
   name: myawesomemodel
   dims: [32, 64, 128]
   dropout: 0.2
   # Any other config your model uses
   ```

4. **Run**
   ```bash
   python train.py model=myawesomemodel
   ```

---

## 🎛️ Hyperparameter Tuning (Grid & Randomized Search)

The framework integrates natively with Hydra's multirun and Optuna swept capabilities. Before running a full LOSO evaluation, you can identify high-impact parameters like `lr`, `batch_size`, and `weight_decay` quickly. 

To tune, use the `-m` (multirun) flag alongside the `hparams_search` configurations, which employs a subject-wise training and validation proxy split (80/20 subject split) instead of iterating over every single fold.

**1. Randomized Search (Optuna/TPE):**
```bash
python train.py -m hparams_search=optuna model=mobilevit dataset=pamap2_dataset strategy=clean
```

**2. Grid Search:**
```bash
python train.py -m hparams_search=grid model=mobilevit
```

You can customize the tuned hyperparameters by editing `conf/hparams_search/optuna.yaml` or `conf/hparams_search/grid.yaml`.

---

## 🧪 Experiments & Strategies
The framework inherently allows creating custom degradation "strategies". You can define strategies in `conf/strategy/` to evaluate your model's robustness against noise, missing channels (modality dropout), or dropped frames. Use them seamlessly via the `strategy=` argument to measure accuracy and F1 scores in non-ideal conditions.
