# Banking Intent Classification with Unsloth

## 1. Overview

This project implements an **intent classification system in the banking domain** using a subset of the BANKING77 dataset. The system is built by fine-tuning a large language model using the **Unsloth framework**, followed by deploying a standalone inference module for real-time prediction.

The pipeline covers the full lifecycle:

- Data preparation
- Model fine-tuning
- Evaluation
- Inference deployment

---

## 2. Project Structure

```
fine-tune-intent-detection-model/
│
├── scripts/
│   ├── preprocess_data.py     # Data preparation and preprocessing
│   ├── train.py               # Fine-tuning script
│   ├── inference.py           # Inference implementation
│
├── configs/
│   ├── train.yaml             # Training configuration
│   ├── inference.yaml         # Inference configuration
│
├── sample_data/
│   ├── train.csv              # Training dataset (subset)
│   ├── test.csv               # Testing dataset
│
├── train.sh                   # Script to run training
├── inference.sh               # Script to run inference
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
```

---

## 3. Setup Environment

### 3.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.2 (Optional) Use GPU environment

Recommended platforms:

- Google Colab
- Kaggle

---

## 4. Dataset Preparation

### 4.1 Source

- Dataset: BANKING77 (from HuggingFace)
- Link: https://huggingface.co/datasets/PolyAI/banking77

### 4.2 Preprocessing

Run:

```bash
python scripts/preprocess_data.py
```

This step will:

- Load the dataset
- Select a subset of intents
- Clean and normalize text
- Map labels to IDs
- Split into train/test sets
- Save processed data to `sample_data/`

---

## 5. Training

### 5.1 Configuration

Edit:

```
configs/train.yaml
```

Key parameters:

- model_name
- batch_size
- learning_rate
- num_epochs
- max_seq_length

### 5.2 Run training

```bash
bash train.sh
```

or

```bash
python scripts/train.py --config configs/train.yaml
```

### 5.3 Output

- Trained model checkpoint
- Logs of training process

---

## 6. Evaluation

The model is evaluated on a held-out test set.

Metrics:

- Accuracy

(Optional improvements):

- Confusion matrix
- Per-class performance analysis

---

## 7. Inference

### 7.1 Run inference

```bash
bash inference.sh
```

or

```bash
python scripts/inference.py --config configs/inference.yaml
```

### 7.2 Usage Example

```python
from scripts.inference import IntentClassification

model = IntentClassification("configs/inference.yaml")

text = "I lost my credit card"
prediction = model(text)

print(prediction)
```

---

## 8. Model Design

### 8.1 Approach

- Task: Text classification (intent detection)
- Method: Fine-tuning LLM with Unsloth
- Training type: Parameter-efficient fine-tuning (e.g., LoRA)

### 8.2 Key Decisions

- Reduced number of intents to fit compute constraints
- Balanced dataset sampling
- Optimized hyperparameters for convergence and stability

---

## 9. Reproducibility

To reproduce results:

1. Run preprocessing
2. Run training with provided config
3. Run inference using saved checkpoint

---

## 10. Demo Video

A demonstration video is available at:

```
[Insert Google Drive link here]
```

The video includes:

- Running inference script
- Example input
- Predicted output
- Final model accuracy

---

## 11. Notes

- Ensure consistent preprocessing between training and inference
- Do not modify dataset structure after training
- Always load model from saved checkpoint (no hardcoding)

---

## 12. Future Improvements

- Use larger subset of intents
- Hyperparameter tuning
- Add validation set
- Deploy as API service

---
