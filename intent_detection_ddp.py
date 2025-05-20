import os
import torch
from datasets import load_dataset, load_from_disk, Dataset
from transformers.trainer_utils import get_last_checkpoint
from augmentation import (
    augment_dataset_with_paraphrasing,
    augment_dataset_with_synonyms,
    augment_dataset_with_btranslation,
    load_combine_datasets,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt


# Load the CLINC150 dataset
clinc_data = load_dataset("contemmcm/clinc150", "full")
if "complete" in clinc_data:
    full_dataset = clinc_data["complete"]
else:
    full_dataset = clinc_data

# Extract intent labels and mappings
unique_intents = full_dataset.features["intent"].names
id2label = full_dataset.features["intent"].int2str
label2id = {label: i for i, label in enumerate(unique_intents)}

# Split the dataset
train_dataset = full_dataset.filter(lambda ex: ex["split"] in ["train", "oos_train"])
val_dataset = full_dataset.filter(lambda ex: ex["split"] in ["val", "oos_val"])
test_dataset = full_dataset.filter(lambda ex: ex["split"] in ["test", "oos_test"])

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# Augment the dataset
augmented_dataset_path = "./tokenize_augmented_dataset"
if os.path.exists(augmented_dataset_path):
    print("Loading augmented dataset from file...")
    augmented_train_ds = load_from_disk(augmented_dataset_path)
else:
    print("Augmenting dataset with paraphrasing...")
    para_augmented_train = augment_dataset_with_paraphrasing(
        train_dataset,
        id2label,
        label_names=unique_intents,
    )
    print("Augmenting dataset with synonyms...")
    syn_ds = augment_dataset_with_synonyms(train_dataset, num_syn_repl=2)
    print("Augmenting dataset with back translations...")
    bt_ds = augment_dataset_with_btranslation(train_dataset, id2label)
    print("Combining augmented datasets...")
    augmented_train_ds = load_combine_datasets()
    augmented_train_ds.save_to_disk(augmented_dataset_path)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)

# Rename and remove unnecessary columns
train_dataset = train_dataset.rename_column("intent", "labels").remove_columns(["text", "domain", "split"])
val_dataset = val_dataset.rename_column("intent", "labels").remove_columns(["text", "domain", "split"])
test_dataset = test_dataset.rename_column("intent", "labels").remove_columns(["text", "domain", "split"])

# Set dataset format for PyTorch
train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

# Model
num_intents = len(unique_intents)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_intents)
model.config.id2label = id2label
model.config.label2id = label2id
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./intent_model",
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    logging_dir="./logs",
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
val_metrics = trainer.evaluate(val_dataset)
print("Validation Set Performance:", val_metrics)
test_metrics = trainer.evaluate(test_dataset)
print("Test Set Performance:", test_metrics)

# Save the model
trainer.save_model("./intent_model")

# OOS Detection
def max_softmax_prob(logits, T=1.0):
    probs = F.softmax(logits / T, dim=-1)
    return float(probs.max())

def energy_score(logits, T=1.0):
    return float(-T * torch.logsumexp(logits / T, dim=-1))

def calibrate_oos_thresholds_roc(trainer, val_dataset, oos_label_name="oos:oos", T=1.0, plot=False):
    pred_output = trainer.predict(val_dataset)
    logits_np = pred_output.predictions
    labels_np = pred_output.label_ids

    label2id = trainer.model.config.label2id
    oos_id = label2id[oos_label_name]

    max_probs = []
    energies = []
    is_oos = []
    for logit_vec, lbl in zip(torch.tensor(logits_np), labels_np):
        max_probs.append(max_softmax_prob(logit_vec, T))
        energies.append(energy_score(logit_vec, T))
        is_oos.append(int(lbl) == oos_id)

    max_probs = np.array(max_probs)
    energies = np.array(energies)
    is_oos = np.array(is_oos, dtype=bool)

    if plot:
        plt.hist(max_probs[~is_oos], bins=50, alpha=0.5, label="in-scope")
        plt.hist(max_probs[is_oos], bins=50, alpha=0.5, label="OOS")
        plt.legend()
        plt.show()

    fpr_p, tpr_p, thr_p = roc_curve(is_oos, -max_probs)
    j_scores_p = tpr_p - fpr_p
    best_idx_p = np.argmax(j_scores_p)
    tau_prob = thr_p[best_idx_p]

    fpr_e, tpr_e, thr_e = roc_curve(is_oos, energies)
    j_scores_e = tpr_e - fpr_e
    best_idx_e = np.argmax(j_scores_e)
    tau_energy = thr_e[best_idx_e]

    return tau_prob, tau_energy

tau_prob, tau_energy = calibrate_oos_thresholds_roc(trainer, val_dataset, plot=True)

def predict_intent_with_oos(text, tau_prob, tau_energy, T=1.0):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    model.eval()
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits[0]

    p = max_softmax_prob(logits, T)
    e = energy_score(logits, T)
    pred_id = int(logits.argmax().item())
    intent = model.config.id2label[pred_id]

    if p < tau_prob or e > tau_energy:
        return {"intent": "oos", "confidence": p, "energy": e}

    return {"intent": intent, "confidence": p, "energy": e}

# Interactive loop for predictions
while True:
    query = input("Enter a query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    prediction = predict_intent_with_oos(query, tau_prob, tau_energy)
    print(f"Query: {query}")
    print(f"Prediction: {prediction}")