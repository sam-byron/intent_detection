import torch
from datasets import load_dataset
import os
from transformers.trainer_utils import get_last_checkpoint

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
    num_train_epochs=30,                # let's fine-tune for 3 epochs (adjustable)
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
    train_dataset=train_dataset,
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

# 1️⃣ Utility functions to compute scores
def max_softmax_prob(logits: torch.Tensor, T: float = 1.0) -> float:
    """
    Returns the maximum softmax probability from `logits / T`.
    """
    scaled = logits / T
    probs = F.softmax(scaled, dim=-1)
    return float(probs.max())

def energy_score(logits: torch.Tensor, T: float = 1.0) -> float:
    """
    Energy score: -T * logsumexp(logits / T)
    Higher values → more likely OOD.
    """
    return float(-T * torch.logsumexp(logits / T, dim=-1))


# 2️⃣ Calibrate thresholds on your validation set
#    We’ll run the model on val_dataset, collect (max_prob, energy) for both in-scope and oos labels,
#    then pick thresholds that best separate them (e.g. midpoint between distributions).

# 2a. Predict on val split
val_out = trainer.predict(val_dataset)
val_logits = torch.from_numpy(val_out.predictions)   # shape (N_val, num_labels)
val_labels = val_out.label_ids                       # shape (N_val,)

# 2b. Identify your OOS label ID
#     (assuming your OOS intent was named exactly "oos" in the original label list)
oos_id = model.config.label2id.get("oos:oos", None)
if oos_id is None:
    raise ValueError("Could not find 'oos' label in model.config.label2id")

# 2c. Compute scores for each example
max_probs = []
energies  = []
is_oos    = []

for logit, lbl in zip(val_logits, val_labels):
    max_probs.append(max_softmax_prob(logit))
    energies.append(energy_score(logit))
    is_oos.append(lbl == oos_id)

max_probs = np.array(max_probs)
energies  = np.array(energies)
is_oos    = np.array(is_oos)

# 2d. Choose thresholds
#    e.g. threshold_prob = midpoint between
#      median(max_probs[~is_oos]) and median(max_probs[is_oos])
#    similarly for energy.
th_prob = 0.5 * (np.median(max_probs[~is_oos]) + np.median(max_probs[is_oos]))
th_energy = 0.5 * (np.median(energies[~is_oos]) + np.median(energies[is_oos]))

print(f"Calibrated softmax‐prob threshold: {th_prob:.3f}")
print(f"Calibrated energy threshold:       {th_energy:.3f}")


# 3️⃣ Augmented inference function with OOS/OOD detection
def predict_intent_with_oos(
    text: str,
    tau_prob: float = th_prob,
    tau_energy: float = th_energy,
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


# Interactive loop for intent prediction
while True:
    example_query = input("Enter a query (or type 'exit' to quit): ")
    if example_query.lower() == "exit":
        print("Exiting...")
        break
    pred_intent = predict_intent_with_oos(example_query)
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
