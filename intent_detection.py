
import torch
from datasets import load_dataset, load_from_disk
import os
from transformers.trainer_utils import get_last_checkpoint
from augmentation import save_augmented_dataset

# Load the CLINC150 dataset from Hugging Face (this will download the data_full version)
# The clinc_data object is a DatasetDict or Dataset containing all 23,700 samples in CLINC150. The dataset includes a column for the user utterance text, an intent label (e.g. "banking:balance"), a broader domain label, and a split indicator. The data is divided into training, validation, and test splits, with out-of-scope examples separated as well (e.g., "oos_train", "oos_test" for out-of-scope queries). 
clinc_data = load_dataset("contemmcm/clinc150", "full")
print(clinc_data)

# The dataset might be stored under a single split "complete", so we filter by the 'split' field:
if "complete" in clinc_data:
    full_dataset = clinc_data["complete"]       # All data
else:
    full_dataset = clinc_data                  # Handle case where directly a Dataset is returned

# Separate into training, validation, and test sets, including out-of-scope (oos) examples
train_dataset = full_dataset.filter(lambda ex: ex["split"] in ["train", "oos_train"])
val_dataset   = full_dataset.filter(lambda ex: ex["split"] in ["val", "oos_val"])
test_dataset  = full_dataset.filter(lambda ex: ex["split"] in ["test", "oos_test"])

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
# Expect ~15100 train (15000 in-scope + 100 oos), 3100 val (3000 + 100), 5500 test (4500 + 1000)


from transformers import AutoTokenizer

# Initialize BERT tokenizer (uncased means text will be lowercased)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function to process text examples
def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
    # max_length=64 should cover most queries; adjust if needed (max utterance length in CLINC150 is relatively short)

# Apply tokenization to each split of the dataset
train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset   = val_dataset.map(tokenize_batch, batched=True)
test_dataset  = test_dataset.map(tokenize_batch, batched=True)

# Rename the intent column to 'labels' for compatibility with the model training API
train_dataset = train_dataset.rename_column("intent", "labels")
val_dataset   = val_dataset.rename_column("intent", "labels")
test_dataset  = test_dataset.rename_column("intent", "labels")

print("Preparing augmented dataset...")
augmented_dataset_path = "./augmented_train_dataset"

if os.path.exists(augmented_dataset_path):
    print("Loading augmented dataset from file...")
    augmented_train = load_from_disk(augmented_dataset_path)
else:
    print("Augmenting dataset...")
    augmented_train = save_augmented_dataset(train_dataset,
                                      num_syn_repl=1,
                                      num_bt=1,
                                      num_synth=3,
                                      output_path=augmented_dataset_path)
# The augment_dataset function will return a new dataset with augmented samples
# Print the number of samples in the augmented training set
print(f"Augmented Train samples: {len(augmented_train)}")

# We can remove other columns we won't use (like 'text', 'domain', 'split') to keep dataset lean
train_dataset = train_dataset.remove_columns(["text", "domain", "split"])
val_dataset   = val_dataset.remove_columns(["text", "domain", "split"])
test_dataset  = test_dataset.remove_columns(["text", "domain", "split"])

# Set formats for PyTorch (so that __getitem__ returns torch.Tensor)
train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")


# Get the list of intent label names (for later use in decoding predictions)
label_names = full_dataset.features["intent"].names  # list of 151 intent labels (e.g., "banking:balance", "oos:oos", etc.)
print(label_names)

from transformers import AutoModelForSequenceClassification

num_intents = len(label_names)  # should be 151 for CLINC150 full
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_intents)

# Set model's label mappings (useful for inference or saving the model)
model.config.id2label = {i: label for i, label in enumerate(label_names)}
model.config.label2id = {label: i for i, label in enumerate(label_names)}

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


from transformers import TrainingArguments, Trainer, IntervalStrategy

training_args = TrainingArguments(
    output_dir="./intent_model",       # output directory for model checkpoints and logs
    overwrite_output_dir=False,
    num_train_epochs=3,                # let's fine-tune for 3 epochs (adjustable)
    per_device_train_batch_size=32,    # batch size for training
    per_device_eval_batch_size=32,     # batch size for evaluation
    learning_rate=2e-5,                # a typical fine-tuning learning rate for BERT
    eval_strategy="epoch",       # evaluate on the validation set each epoch
    # evaluate_during_training=True,    # evaluate during training
    save_strategy="epoch",             # save model each epoch
    load_best_model_at_end=True,       # load best model (according to eval metric) at end of training
    metric_for_best_model="accuracy",  # use accuracy to pick best model (could use f1 as well)
    logging_steps=50,                  # log training progress every 50 steps
    logging_dir="./logs",              # directory for logs
    seed=42,                            # for reproducibility
)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=augmented_train,  # use the augmented training dataset
    eval_dataset=val_dataset,             # validation set for evaluation
    tokenizer=tokenizer,                  # tokenizer is passed to enable automatic padding in the collator
    compute_metrics=compute_metrics       # function to compute metrics
)

trainer.train(get_last_checkpoint(training_args.output_dir))  # start training
# Save the fine-tuned model
# trainer.save_model("./intent_model")  # save the model to the specified directory

# Evaluate the fine-tuned model on the test set
test_metrics = trainer.evaluate(test_dataset)
print("Test Set Performance:", test_metrics)


import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt

def max_softmax_prob(logits: torch.Tensor, T: float = 1.0) -> float:
    probs = F.softmax(logits / T, dim=-1)
    return float(probs.max())

def energy_score(logits: torch.Tensor, T: float = 1.0) -> float:
    return float(-T * torch.logsumexp(logits / T, dim=-1))

def calibrate_oos_thresholds_roc(
    trainer,
    val_dataset,
    oos_label_name: str = "oos:oos",
    T: float = 1.0,
    plot: bool = False
):
    # 1. Gather logits & labels
    pred_output = trainer.predict(val_dataset)
    logits_np   = pred_output.predictions    # (N, num_labels)
    labels_np   = pred_output.label_ids      # (N,)

    # 2. Identify OOS ID
    label2id = trainer.model.config.label2id
    if oos_label_name not in label2id:
        raise ValueError(f"Label '{oos_label_name}' not found.")
    oos_id = label2id[oos_label_name]

    # 3. Compute scores & ground-truth
    max_probs = []
    energies  = []
    is_oos    = []
    for logit_vec, lbl in zip(torch.tensor(logits_np), labels_np):
        max_probs.append(max_softmax_prob(logit_vec, T))
        energies.append(energy_score(logit_vec, T))
        is_oos.append(int(lbl) == oos_id)

    max_probs = np.array(max_probs)
    energies  = np.array(energies)
    is_oos    = np.array(is_oos, dtype=bool)

    # Optional: visualize
    if plot:
        plt.hist(max_probs[~is_oos], bins=50, alpha=0.5, label="in-scope")
        plt.hist(max_probs[is_oos],  bins=50, alpha=0.5, label="OOS")
        plt.title("Softmax Max-Prob Distributions"); plt.legend(); plt.show()

        plt.hist(energies[~is_oos], bins=50, alpha=0.5, label="in-scope")
        plt.hist(energies[is_oos],  bins=50, alpha=0.5, label="OOS")
        plt.title("Energy Score Distributions"); plt.legend(); plt.show()

    # 4. Find best softmax-prob threshold via Youden’s J
    #    We invert max_probs so that higher “score” => more likely OOS
    fpr_p, tpr_p, thr_p = roc_curve(is_oos, -max_probs)
    j_scores_p = tpr_p - fpr_p
    best_idx_p = np.argmax(j_scores_p)
    tau_prob   = thr_p[best_idx_p]

    # 5. Find best energy threshold via Youden’s J
    fpr_e, tpr_e, thr_e = roc_curve(is_oos, energies)
    j_scores_e = tpr_e - fpr_e
    best_idx_e = np.argmax(j_scores_e)
    tau_energy = thr_e[best_idx_e]

    print(f"→ Softmax-prob τₚ = {tau_prob:.4f} (J={j_scores_p[best_idx_p]:.3f})")
    print(f"→ Energy τ_E        = {tau_energy:.3f} (J={j_scores_e[best_idx_e]:.3f})")

    # 6. Evaluate combined detector
    preds_oos = (max_probs < tau_prob) | (energies > tau_energy)
    tn, fp, fn, tp = confusion_matrix(is_oos, preds_oos).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("Combined Detector — Confusion Matrix:")
    print(f"    True In-Scope   as In-Scope:  TN={tn}")
    print(f"    True In-Scope   as OOS:       FP={fp}")
    print(f"    True OOS        as In-Scope:  FN={fn}")
    print(f"    True OOS        as OOS:       TP={tp}")
    print(f"Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    return tau_prob, tau_energy


# 3️⃣ Augmented inference function with OOS/OOD detection
def predict_intent_with_oos(
    text: str,
    tau_prob: float,
    tau_energy: float,
    T: float = 1.0
) -> dict:
    """
    Tokenize `text`, run the model, and:
      - compute max softmax prob & energy,
      - if either indicates OOD, return 'oos',
      - otherwise return the predicted in-scope intent.
    Returns a dict with:
      * intent: str
      * confidence: max softmax prob
      * energy: float
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    model.eval()
    with torch.no_grad():
        inputs.to(device)  # Move inputs to the same device as the model
        logits = model(**inputs).logits[0]

    p   = max_softmax_prob(logits, T)
    e   = energy_score(logits, T)
    pred_id = int(logits.argmax().item())
    intent = model.config.id2label[pred_id]

    if p < tau_prob or e > tau_energy:
        return {"intent": "oos", "confidence": p, "energy": e}

    return {"intent": intent, "confidence": p, "energy": e}

# ─── Usage ────────────────────────────────────────────────────────────────────
# After you have trained your model and have a Trainer instance:

tau_prob, tau_energy = calibrate_oos_thresholds_roc(
    trainer,
    val_dataset,
    oos_label_name="oos:oos",
    T=1.0,
    plot=True       # toggle to see the histograms
)

# Interactive loop for intent prediction
while True:
    example_query = input("Enter a query (or type 'exit' to quit): ")
    if example_query.lower() == "exit":
        print("Exiting...")
        break
    # The following number were chose by analyzing the ROC curve and the energy score distributions
    pred_intent = predict_intent_with_oos(example_query, 0.18, -5.9, 1.0)
    # pred_intent = predict_intent_with_oos(example_query, tau_prob, tau_energy, 1.1)
    print(f"Query: '{example_query}'")
    print(f"Predicted Intent: {pred_intent}")

# Enter a query (or type 'exit' to quit): I want to travel to india
# Query: 'I want to travel to india'
# Predicted Intent: travel:international_visa
# Enter a query (or type 'exit' to quit): Book me an appointment with the dentist
# Query: 'Book me an appointment with the dentist'
# Predicted Intent: auto_and_commute:schedule_maintenance
# Enter a query (or type 'exit' to quit): Book me an appointment with the dentist
# Query: 'Book me an appointment with the dentist'
# Predicted Intent: auto_and_commute:schedule_maintenance
# Enter a query (or type 'exit' to quit): My son has been become dangerously sick. I need to tend to him for the next week and be away from work.
# Query: 'My son has been become dangerously sick. I need to tend to him for the next week and be away from work.'
# Predicted Intent: work:pto_used
# Enter a query (or type 'exit' to quit): My son has been become dangerously sick. I need time off.
# Query: 'My son has been become dangerously sick. I need time off.'
# Predicted Intent: work:pto_request_status
