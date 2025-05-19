import torch
from datasets import load_dataset, load_from_disk, Dataset
import os
from transformers.trainer_utils import get_last_checkpoint
from augmentation import augment_dataset_with_paraphrasing, augment_dataset_with_synonyms, augment_dataset_with_btranslation, save_load_combine_tokenize_datasets
from modeling_oos import calibrate_oos_thresholds_roc, predict_intent_with_oos

# ─── Load the datasets ──────────────────────────────────────────────────────
# Load the CLINC150 dataset
combined_ds, full_dataset, train_dataset, val_dataset, test_dataset, tokenized_train_ds, tokenizer = save_load_combine_tokenize_datasets()

from transformers import AutoModelForSequenceClassification

# PREPARE MODEL
# Get the list of intent label names (for later use in decoding predictions)
label_names = full_dataset.features["intent"].names  # list of 151 intent labels 
num_intents = len(label_names)  # should be 151 for CLINC150 full
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_intents)

# Set model's label mappings (useful for inference or saving the model)
model.config.id2label = {i: label for i, label in enumerate(label_names)}
model.config.label2id = {label: i for i, label in enumerate(label_names)}

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# TRAIN MODEL

from transformers import TrainingArguments, Trainer, IntervalStrategy

training_args = TrainingArguments(
    output_dir="./intent_model",       # output directory for model checkpoints and logs
    overwrite_output_dir=False,
    num_train_epochs=6,                # let's fine-tune for 3 epochs (adjustable)
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
    train_dataset=tokenized_train_ds,  # use the augmented training dataset
    eval_dataset=val_dataset,             # validation set for evaluation
    tokenizer=tokenizer,                  # tokenizer is passed to enable automatic padding in the collator
    compute_metrics=compute_metrics       # function to compute metrics
)

# trainer.train(get_last_checkpoint(training_args.output_dir))  # start training
trainer.train()  # start training
# Evaluate the model on the validation set
val_metrics = trainer.evaluate(val_dataset)
print("Validation Set Performance:", val_metrics)   
# Save the fine-tuned model
# trainer.save_model("./intent_model")  # save the model to the specified directory

# Evaluate the fine-tuned model on the test set
test_metrics = trainer.evaluate(test_dataset)
print("Test Set Performance:", test_metrics)




# ─── Usage ────────────────────────────────────────────────────────────────────
# After you have trained your model and have a Trainer instance:

tau_prob, tau_energy = calibrate_oos_thresholds_roc(
    trainer,
    val_dataset,
    model,
    tokenizer,
    device,
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
    # pred_intent = predict_intent_with_oos(example_query, tau_prob, tau_energy, 1.0)
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