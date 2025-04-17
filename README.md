<div align="center">
    <img src="https://github.com/HarunoriKawano/speaker-identification-with-tgp/blob/main/docs/title.png" width="800px">
</div>

<br/>
 <div align="center">
    <a href="https://github.com/pytorch/pytorch">
        <img src="https://img.shields.io/badge/framework-PyTorch-red"> 
    </a>
    <a href="https://github.com/HarunoriKawano/best-rq/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache--2.0-informational"> 
    </a>
    <a href="https://www.python.org/dev/peps/pep-0008/">
        <img src="https://img.shields.io/badge/codestyle-PEP--8-informational"> 
    </a>
    <a href="https://github.com/HarunoriKawano/best-rq">
        <img src="https://img.shields.io/badge/build-passing-success"> 
    </a>
</div>

***

## Overview
<div align="center">
    <img src="https://github.com/HarunoriKawano/speaker-identification-with-tgp/blob/main/docs/overview.png" width="600px" >
</div>


# 🎙️ Speaker Identification with Temporal Gate Pooling (TGP)

This repository implements a Conformer-based speaker identification model, enhanced with **Temporal Gate Pooling (TGP)**. It supports multiple pooling strategies and is trained on a subset of the VoxCeleb dataset.

## Reference Paper
An Effective Transformer-based Contextual Model and Temporal Gate Pooling for Speaker Identification
[Link Text](https://arxiv.org/abs/2308.11241)

---

## 📌 Features

- ✅ **Conformer Encoder**: Captures both local and global acoustic features.
- ✅ **Temporal Gate Pooling (TGP)**: Core method to aggregate temporal features efficiently.
- ✅ **Multiple Pooling Support**: Statistical, Self-Attention, Temporal Gate.
- ✅ **Speaker Classifier**: Final embedding projection and speaker classification.
- ✅ **Evaluation**: Accuracy calculation and training logs.
- ✅ **VoxCeleb-Compatible**: Designed to train/evaluate on VoxCeleb subsets.

---

## 🧠 Methodology Overview

### 🔄 Preprocessing
- **Fixed-length padding/truncation**
- **Mel-spectrogram extraction** using `torchaudio.transforms.MelSpectrogram`

### 🧱 Model Architecture
1. **WavToMel**: Converts raw audio into log-Mel spectrogram.
2. **ConvSubsampling**: Reduces time resolution while increasing feature dimensionality.
3. **Conformer Encoder**: Series of Conformer blocks for rich acoustic modeling.
4. **Pooling Layer**:
   - `temporal_gate` (TGP)
   - `self-attention`
   - `mean`, `mean_std`, `max`, `random`
5. **FC Layer**: Projects pooled representation to an embedding space.
6. **Classifier**: Classifies the embedding into speaker identity.

---

## 🧪 Requirements

```bash
pip install -r requirements.txt
```

---

## 📁 Folder Structure

```
├── train.py                     # Training script
├── eval.py                      # Evaluation script
├── config/train.yml             # Main configuration file
├── speaker_identification/      # Model components
├── conformer/                   # Encoder & subsampling modules
├── pooling_layer/               # TGP, Self-attention, Statistical
├── pre_processing/              # Mel-spectrogram + padding logic
├── utils/                       # Dataset & samplers
├── checkpoints/                 # Saved models
```

---

## ⚙️ Training

Update the config file at `config/train.yml` and run:

```bash
python train.py --config config/train.yml
```

Training loss and validation accuracy will be logged in:

- `checkpoints/loss_log.txt`
- `checkpoints/acc_log.txt`

---

## 🧾 Evaluation

Evaluate a saved model with:

```bash
python eval.py --config config/train.yml --model_path checkpoints/best_model.pth
```

You will get:
- Overall **accuracy**
- Per-class prediction (optional for extension)

---

## 🔧 Configuration Highlights (`train.yml`)

```yaml
model:
  encoder_type: conformer
  pooling_type: temporal_gate   # or: self-attention, mean, etc.
  pooling_num_heads: 4
  filter_size: 512
  hidden_size: 128
  num_layers: 4
  ...
training:
  batch_size: 16
  epochs: 1
  learning_rate: 1e-4
```

---

## 🚀 Future Work (Optional)

- 🔁 Add support for BEST-RQ pretraining
- 📈 Add TensorBoard logger
- 🧪 Pooling comparison ablation
- 📦 Push results to WandB / HuggingFace

---

## ✍️ Credits

This implementation is based on a simplified architecture adapted from the original [TGP paper](https://arxiv.org/abs/2206.07993). All code is organized for educational and experimental use.

---

## 📄 License

MIT License
