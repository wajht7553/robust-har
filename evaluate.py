import os
import argparse
import yaml
import torch
import json
from src.models.factory import create_model
from src.data.splitter import LOSOSplitter, create_dataloaders
from src.data.transforms import SensorFailureTransform
from src.training.trainer import Trainer
from src.utils.metrics import compute_metrics
from src.utils.common import load_model


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_scenario(model, test_loader, device, scenario_name):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())  # Dummy
    trainer = Trainer(model, device, criterion, optimizer)

    print(f"Evaluating scenario: {scenario_name}")
    _, _, y_pred, y_true = trainer.validate(test_loader)
    metrics = compute_metrics(y_true, y_pred)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", type=str, required=True, help="Path to experiment directory"
    )
    parser.add_argument(
        "--subject", type=str, default="proband1", help="Subject to evaluate"
    )
    parser.add_argument("--data_dir", type=str, default="dataset/processed_acc_gyr")
    args = parser.parse_args()

    # Load experiment info
    with open(os.path.join(args.experiment_dir, "results.json"), "r") as f:
        results = json.load(f)

    with open(os.path.join(args.experiment_dir, "model_config.json"), "r") as f:
        model_config = json.load(f)

    model_name = results["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    splitter = LOSOSplitter(args.data_dir)
    X_train, y_train, X_test, y_test = splitter.get_train_test_split(args.subject)

    # Check if input channels were filtered during training
    # This logic should ideally be more robust (saving input_channels in config)
    # For now, we check if model_config['nb_channels'] matches data
    if model_config["nb_channels"] < X_test.shape[2]:
        print(
            f"Model expects {model_config['nb_channels']} channels, data has {X_test.shape[2]}. Truncating..."
        )
        X_train = X_train[:, :, : model_config["nb_channels"]]
        X_test = X_test[:, :, : model_config["nb_channels"]]

    # Load model
    model = create_model(model_name, model_config)
    model_path = os.path.join(args.experiment_dir, f"model_{args.subject}.pt")
    load_model(model, model_path, device)

    # Scenarios
    scenarios = {
        "Clean": None,
        "Missing Gyro": SensorFailureTransform(p_dropout_gyro=1.0),
        "Noisy Accel (std=0.1)": SensorFailureTransform(p_noise=1.0, noise_std=0.1),
        "Random Channel Drop (p=0.2)": SensorFailureTransform(p_channel_drop=0.2),
    }

    scenario_results = {}
    for name, transform in scenarios.items():
        _, test_loader, _ = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=32, test_transform=transform
        )
        metrics = evaluate_scenario(model, test_loader, device, name)
        scenario_results[name] = metrics
        print(f"{name}: Acc={metrics['accuracy']:.4f}")

    # Save report
    with open(
        os.path.join(args.experiment_dir, f"robustness_report_{args.subject}.json"), "w"
    ) as f:
        json.dump(scenario_results, f, indent=2)
    print(f"Report saved to {args.experiment_dir}")


if __name__ == "__main__":
    main()
