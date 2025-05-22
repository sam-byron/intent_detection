import torch
from datasets import load_dataset, load_from_disk, Dataset
import os
from transformers.trainer_utils import get_last_checkpoint
from augmentation import save_load_combine_tokenize_datasets
from modeling_oos import calibrate_oos_thresholds_roc, predict_intent_with_oos

model_name = "bert-large-uncased-whole-word-masking"

# ─── Load the datasets ──────────────────────────────────────────────────────
# Load the CLINC150 dataset
combined_ds, full_dataset, train_dataset, val_dataset, test_dataset, oos_ds, tokenizer = save_load_combine_tokenize_datasets()

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize BERT tokenizer (uncased means text will be lowercased)
# The texts are lowercased and tokenized using WordPiece and a vocabulary size of 30,000. The inputs of the model are then of the form:
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # NOTE: THIS WAS ORIGINALLY USED
# tokenizer.pad_token = tokenizer.eos_token
#  # tell the tokenizer what its max context really is
# tokenizer.model_max_length = config["n_positions"]

# Tokenization function to process text examples
def tokenize_batch(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=64)
    # max_length=64 should cover most queries; adjust if needed (max utterance length in CLINC150 is relatively short)

# Apply tokenization to each split of the dataset
tokenized_val_dataset = val_dataset.map(tokenize_batch, batched=True)
tokenized_train_ds = train_dataset.map(tokenize_batch, batched=True)

tokenized_test_dataset  = test_dataset.map(tokenize_batch, batched=True)
tokenized_combined_ds = combined_ds.map(tokenize_batch, batched=True)
tokenized_oos_ds = oos_ds.map(tokenize_batch, batched=True)


# PREPARE MODEL
# Get the list of intent label names (for later use in decoding predictions)
label_names = full_dataset.features["intent"].names  # list of 151 intent labels 
num_intents = len(label_names)  # should be 151 for CLINC150 full
model_dir = "./intent_model"  # directory where the model is saved
# last_checkpoint = get_last_checkpoint(model_dir)
if os.path.exists(model_dir) and get_last_checkpoint(model_dir) is not None:
    print(f"Loading model from {get_last_checkpoint(model_dir)}")
    model = AutoModelForSequenceClassification.from_pretrained(
        get_last_checkpoint(model_dir),
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.2,
        classifier_dropout=0.2
    )
else:
    print("Loading pretrained BERT model")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_intents
    )

# Set model's label mappings (useful for inference or saving the model)
model.config.id2label = {i: label for i, label in enumerate(label_names)}
model.config.label2id = {label: i for i, label in enumerate(label_names)}

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# TRAIN MODEL

from transformers import TrainingArguments, Trainer, IntervalStrategy
# trainer.train(get_last_checkpoint(training_args.output_dir))  # start training
# output_dir="./intent_model"
training_args = TrainingArguments(
    output_dir=model_dir,       # output directory for model checkpoints and logs
    overwrite_output_dir=False,
    num_train_epochs=3,                # let's fine-tune for 3 epochs (adjustable)
    per_device_train_batch_size=96,    # batch size for training
    per_device_eval_batch_size=96,     # batch size for evaluation
    learning_rate=2e-5,                # a typical fine-tuning learning rate for BERT
    # learning_rate=2e-6,
    eval_strategy="steps",       # evaluate on the validation set each epoch
    eval_steps = 1000,
    # evaluate_during_training=True,    # evaluate during training
    save_strategy="steps",             # save model each epoch
    save_steps=1000,                   # save model every 1000 steps
    load_best_model_at_end=True,       # load best model (according to eval metric) at end of training
    metric_for_best_model="accuracy",  # use accuracy to pick best model (could use f1 as well)
    logging_steps=250,                  # log training progress every 50 steps
    logging_dir="./logs",  
    # ↑ weight decay to penalize large weights more heavily
    weight_decay=0.05,                    
    # ↑ label smoothing to avoid overconfident predictions
    label_smoothing_factor=0.1,                 # directory for logs
    # seed=42,                            # for reproducibility
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
    train_dataset=tokenized_combined_ds,  # use the augmented training dataset
    eval_dataset=val_dataset,             # validation set for evaluation
    tokenizer=tokenizer,                  # tokenizer is passed to enable automatic padding in the collator
    compute_metrics=compute_metrics       # function to compute metrics
)

# trainer.train(get_last_checkpoint(training_args.output_dir))  # start training
if os.path.exists(model_dir) and False:
    trainer.train()  # start training
    # trainer.train(get_last_checkpoint(training_args.output_dir))  # start training
    # Evaluate the model on the validation set
    val_metrics = trainer.evaluate(val_dataset)
    print("Validation Set Performance:", val_metrics)   
    
    # Evaluate the fine-tuned model on the test set
    test_metrics = trainer.evaluate(test_dataset)
    print("Test Set Performance:", test_metrics)

# ─── Usage ────────────────────────────────────────────────────────────────────
# After you have trained your model and have a Trainer instance:

# Fine tune on oos dataset
# # Combine train_ds and oos_ds
oos_train_ds = {"text": [], "labels": []}
oos_train_ds["text"] = tokenized_train_ds["text"] + tokenized_oos_ds["text"]
oos_train_ds["labels"] = tokenized_train_ds["labels"] + tokenized_oos_ds["labels"]
oos_train_ds = Dataset.from_dict(oos_train_ds)
tok_oos_train_ds = oos_train_ds.map(tokenize_batch, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_oos_train_ds,  # use the augmented training dataset
    eval_dataset=val_dataset,             # validation set for evaluation
    tokenizer=tokenizer,                  # tokenizer is passed to enable automatic padding in the collator
    compute_metrics=compute_metrics       # function to compute metrics
)
trainer.train()  # start training

# tau_prob, tau_energy = calibrate_oos_thresholds_roc(
#     trainer,
#     val_dataset,
#     oos_label_name="oos:oos",
#     T=1.0,
#     plot=True       # toggle to see the histograms
# )

tau_prob, tau_energy = calibrate_oos_thresholds_roc(
    trainer,
    tokenized_test_dataset,
    oos_label_name="oos:oos",
    T=1.0,
    plot=True       # toggle to see the histograms
)

# tau_prob, tau_energy = calibrate_oos_thresholds_roc(
#     trainer,
#     tokenized_combined_ds,
#     oos_label_name="oos:oos",
#     T=1.0,
#     plot=True       # toggle to see the histograms
# )

# tau_prob, tau_energy = calibrate_oos_thresholds_roc(
#     trainer,
#     tokenized_train_dataset,
#     oos_label_name="oos:oos",
#     T=1.0,
#     plot=True       # toggle to see the histograms
# )

print(f"tau_prob: {tau_prob}, tau_energy: {tau_energy}")
# The following number were chose by analyzing the ROC curve and the energy score distributions

# Interactive loop for intent prediction
# while True:
#     example_query = input("Enter a query (or type 'exit' to quit): ")
#     if example_query.lower() == "exit":
#         print("Exiting...")
#         break
#     # The following number were chose by analyzing the ROC curve and the energy score distributions
#     # pred_intent = predict_intent_with_oos(example_query, 0.18, -5.9, tokenizer,
#     # model,
#     # device)
#     pred_intent = predict_intent_with_oos(example_query, tau_prob, tau_energy, tokenizer,
#     model,
#     device)
#     print(f"Query: '{example_query}'")
#     print(f"Predicted Intent: {pred_intent}")

# Take the number of samples as input
num_samples = int(input("Enter the number of sample queries to test: "))

# Sample queries from CLINC150 dataset
# Sample queries from the full dataset
sample_queries = [ex for ex in full_dataset["text"]]

# Limit the number of queries based on user input
sample_queries = sample_queries[:num_samples]

for query in sample_queries:
    pred_intent = predict_intent_with_oos(query, tau_prob, tau_energy, tokenizer, model, device)
    print(f"Query: '{query}'")
    print(f"Predicted Intent: {pred_intent}")
    print("-" * 50)