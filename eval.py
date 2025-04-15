import os
import torch
import yaml
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from pre_processing import PreProcessing
from speaker_identification.speaker_identification import SpeakerIdentificationModel
from utils.dataset import SpeakerDataset
from utils.sampler import FixedLengthBatchSampler

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            wavs, lengths, lbls = [x.to(device) for x in batch]
            outputs = model(wavs, lengths)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    return accuracy_score(labels, preds)

def main(config_path, model_path):
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load preprocessor and dataset
    preprocessor = PreProcessing(cfg)
    dataset = SpeakerDataset(cfg, preprocessor)

    test_loader = DataLoader(
        dataset.test_dataset,
        batch_sampler=FixedLengthBatchSampler(dataset.test_dataset, cfg["training"]["batch_size"]),
        collate_fn=dataset.collate_fn,
        num_workers=2
    )

    # Load model
    model = SpeakerIdentificationModel(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")

    # Evaluate
    accuracy = evaluate(model, test_loader, device)
    print(f"🎯 Evaluation Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to model .pth file")
    args = parser.parse_args()

    main(args.config, args.model_path)
