# augmentation.py
import random
import nltk
from nltk.corpus import wordnet
import torch
from datasets import Dataset, disable_caching, load_dataset
from transformers import (
    MarianMTModel, MarianTokenizer, AutoTokenizer,
    pipeline as hf_pipeline,
)
import os
from datasets import concatenate_datasets
import multiprocessing
from multiprocessing import Pool, cpu_count
import torch.distributed as dist
from tqdm import tqdm  # Import tqdm for progress bars

device = "cuda" if torch.cuda.is_available() else "cpu"

# Disable datasets caching
# disable_caching()

# ─── 1. SYNONYM REPLACEMENT ────────────────────────────────────────────────────
nltk.download('wordnet')
nltk.download('omw-1.4')

def synonym_replacement(sentence: str, n: int = 2) -> str:
    """
    Replace up to `n` words in `sentence` with a random WordNet synonym.
    """
    words = sentence.split()
    candidates = [w for w in words if wordnet.synsets(w)]
    if not candidates:
        return sentence  # Return the original sentence if no synonyms are found
    random.shuffle(candidates)
    num_repl = 0
    new_words = words.copy()

    for w in candidates:
        synsets = wordnet.synsets(w)
        lemmas = [l.name().replace('_',' ') for s in synsets for l in s.lemmas()]
        lemmas = [l for l in set(lemmas) if l.lower() != w.lower()]
        if not lemmas:
            continue
        replacement = random.choice(lemmas)
        new_words = [replacement if x==w else x for x in new_words]
        num_repl += 1
        if num_repl >= n:
            break

    return " ".join(new_words)


# ─── 2. BACK-TRANSLATION ───────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use MarianMT models for translation
mt_en_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mt_en_es_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es").to(device)
mt_es_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
mt_es_en_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en").to(device)

def back_translate_batch(batch):
    # English → Spanish
    en_es_tok = mt_en_es_tokenizer(batch["text"], return_tensors="pt", truncation=True, max_length=64, padding=True)
    en_es_tok = {k: v.to(device) for k, v in en_es_tok.items()}  # Move tensors to the correct device
    # batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        en_es_translated = mt_en_es_model.generate(**en_es_tok, num_beams=3, max_length=64)
    # Decode the translated text
    es_texts = mt_en_es_tokenizer.batch_decode(en_es_translated, skip_special_tokens=True)
    es = mt_en_es_tokenizer.batch_decode(en_es_translated, skip_special_tokens=True)[0]
    # Spanish → English
    # Spanish → English
    es_en_tok = mt_es_en_tokenizer(
        es_texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True,  # Add padding here as well
        max_length=64
    )
    es_en_tok = {k: v.to(device) for k, v in es_en_tok.items()}  # Move tensors to the correct devic
    # batch_rev = {k: v.to(device) for k, v in batch_rev.items()}
    with torch.no_grad():
        en_es_translated = mt_es_en_model.generate(**es_en_tok, num_beams=3, max_length=64)
    # Decode the back-translated text
    en_texts = mt_es_en_tokenizer.batch_decode(en_es_translated, skip_special_tokens=True)
    
    # Return a dictionary with the updated "text" column
    return {"text": en_texts}

import random
import torch
from datasets import Dataset
from transformers import pipeline

# ─── 0. Prepare paraphraser ───────────────────────────────────────────────────
# Use an instruction-tuned T5 model for high-quality paraphrases.
paraphraser = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",      # or "google/flan-t5-base" if you need smaller
    tokenizer="google/flan-t5-large",
    device=device,
    # you can tweak generation settings:
    max_length=64,
    do_sample=True,
    # top_p=0.9,
    # temperature=0.8
)

# ─── 1. Paraphrase helper ─────────────────────────────────────────────────────
def paraphrase(text, num_return_sequences: int = 3) -> list[str]:
    """Return up to `num_return_sequences` paraphrases of `text`."""
    # prompts = []
    # for text in texts:
    #     prompts.append(f"paraphrase: {text} </s>")
    prompt = f"paraphrase: {text} </s>"
    prompt.to(device)
    paraphraser.to(device)
    paraphraser.no_grad()
    outs = paraphraser(prompt, num_return_sequences=num_return_sequences)
    # print(f"Paraphrased [{text}] to {len(outs)} candidates.")
    # print(outs)
    return [out["generated_text"].strip() for out in outs]

def paraphrase_batch(batch: dict, num_return_sequences: int = 1, paraph_per_seed: int = 2) -> dict:
    """
    Return a dictionary with paraphrased texts for each text in the batch.
    """
    # Prepare prompts for all texts in the batch
    prompts = [f"paraphrase: {text} </s>" for text in batch["text"] for _ in range(paraph_per_seed)]
    # outs = paraphraser(prompts, num_beams = 3, num_return_sequences=num_return_sequences, batch_size=128)
    outs = paraphraser(prompts, num_beams = 3, batch_size=128)

    # Extract the generated_text values from the output
    paraphrased_texts = [x["generated_text"].strip() for x in outs]

    # Return a dictionary with the updated "text" column
    return {"text": paraphrased_texts}

# ─── 2. Synthetic generator ───────────────────────────────────────────────────

def generate_synthetic_for_intents(
    train_ds, 
    id2label, 
    k: int = 5,
    paraph_per_seed: int = 2
) -> dict:
    """
    Generate synthetic examples for a given intent using paraphrasing.
    """
    # Wrap each text in a dictionary
    samples = [{"text": ex["text"]} for ex in train_ds]

    # Convert samples to a Hugging Face Dataset
    samples = Dataset.from_list(samples)

    # Apply paraphrasing in batches
    paras_samples = samples.map(
        lambda batch: paraphrase_batch(batch, num_return_sequences=paraph_per_seed),
        batched=True,
        batch_size=128
    )

    # Extract intents
    intents = [
        id2label(ex["intent"]) for ex in train_ds for _ in range(paraph_per_seed)
    ]

    # Return paraphrases and intents
    return {"paraphrases": paras_samples["text"], "intents": intents}



# ─── 4. Example usage ─────────────────────────────────────────────────────────
# Suppose train_dataset has your original CLINC150 training split
# unique_intents = list({train_dataset.features["labels"].int2str(l) for l in train_dataset["labels"]})
# Or pick a subset you want to augment first:
# unique_intents = ["banking:balance", "travel:flight_status", ...]

# augmented_train = augment_dataset_with_paraphrases(
#     train_ds=train_dataset,
#     intents=unique_intents,
#     k_per_intent=8
# )

# Now feed `augmented_train` into your Trainer.


# Load the tokenizer for the model we will use for tokenization
# (e.g., BERT, RoBERTa, etc.)
# This should match the model you will use for training.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(batch, padding="max_length", truncation=True, max_length=64)


def augment_dataset_with_paraphrasing(train_ds, id2label, label_names, output_path="./paraphrase_aug_dataset"):
    """
    Augment the dataset using paraphrasing, synonym translaton, and back-translation pipelines in batches.
    """
    
    # Generate synthetic texts for this intent
    # print(f"Generating synthetic text for intent: {intent}\n")
    synth_options = generate_synthetic_for_intents(train_ds, id2label)

    augmented_ds = {"text": synth_options["paraphrases"], "labels": synth_options["intents"]}

    print("Paraphrasing dataset created.")

    paraphrases_ds = Dataset.from_dict(synth_options)
    # Save the combined dataset to disk
    os.makedirs(output_path, exist_ok=True)
    print("Saving augmented dataset to file...")
    paraphrases_ds.save_to_disk(output_path)
    # tokenized = Dataset.from_list(tokenized['input_ids'])
    # tokenized.save_to_disk(output_paugment_dataset_with_btranslationath)

    return paraphrases_ds


def augment_dataset_with_synonyms(train_ds, num_syn_repl=1, output_path="./synonyms_aug_dataset"):
    id2label = train_ds.features["intent"].int2str
    augmented_texts = []
    augmented_labels = []
    # We assume the original batch still has a "text" field.
    print("Generating synonyms dataset...")
    for text, label in zip(train_ds["text"], train_ds["intent"]):
        # Apply synonym replacement
        synonyms_text = synonym_replacement(text, n=num_syn_repl)
        augmented_texts.append(synonyms_text)
        augmented_labels.append(id2label(label))
    syn_ds = Dataset.from_dict({"text": augmented_texts, "labels": augmented_labels})
    syn_ds.save_to_disk(output_path)
    print("Synonym replacements dataset saved.")
    # Return with the same keys as expected by the tokenizer later
    return syn_ds

def augment_dataset_with_btranslation(train_ds, id2label, num_bt=2, output_path="./backtranslation_aug_dataset"):
    # id2label = train_ds.features["intent"].int2str
    augmented_texts = []
    augmented_labels = []
    bt_ds = train_ds.map(back_translate_batch, batched=True, batch_size=128)
    # Save the back-translated dataset to disk
    os.makedirs(output_path, exist_ok=True)
    print("Saving back-translated dataset to file...")
    # Return with the same keys as expected by the tokenizer later

    # convert all values of key intent to label using id2label
    # Convert the intent column from integer IDs to string labels
    # bt_ds = bt_ds.map(lambda x: {"intent": id2label(x["intent"])}, batched=False)

    # Extract intents
    intents = [
        id2label(ex) for ex in bt_ds["intent"]
    ]

    bt_ds = Dataset.from_dict({"text": bt_ds["text"], "intents": intents})
    bt_ds.save_to_disk(output_path)


def load_combine_datasets(par_output_path="./paraphrase_aug_dataset", 
                     syn_output_path="./synonyms_aug_dataset", 
                     bt_output_path="./backtranslation_aug_dataset"):
    
    # Load the datasets from disk
    par_ds = Dataset.load_from_disk(par_output_path)
    syn_ds = Dataset.load_from_disk(syn_output_path)
    bt_ds = Dataset.load_from_disk(bt_output_path)

    clinc_data = load_dataset("contemmcm/clinc150", "full")
    print(clinc_data)

    # The dataset might be stored under a single split "complete", so we filter by the 'split' field:
    if "complete" in clinc_data:
        full_dataset = clinc_data["complete"]       # All data
    else:
        full_dataset = clinc_data                  # Handle case where directly a Dataset is returned

    # Separate into training, validation, and test sets, including out-of-scope (oos) examples
    train_ds = full_dataset.filter(lambda ex: ex["split"] in ["train", "oos_train"])

    # Rename the intent column to 'labels' for compatibility with the model training API
    par_ds = par_ds.rename_column("paraphrases", "text")
    par_ds = par_ds.rename_column("intents", "intent")
    bt_ds = bt_ds.rename_column("intents", "intent")
    syn_ds = syn_ds.rename_column("labels", "intent")
    # train_ds = train_ds.rename_column("intents", "intent")

    # We can remove other columns we won't use (like 'text', 'domain', 'split') to keep dataset lean
    # bt_ds = bt_ds.remove_columns(["domain", "split"])

    train_ds = train_ds.remove_columns(["domain", "split"])
    id2label = full_dataset.features["intent"].int2str
     # Extract intents
    intent = [
        id2label(ex) for ex in train_ds["intent"]
    ]
    train_ds = Dataset.from_dict({"text": train_ds["text"], "intent": intent})

    # Combine the datasets
    combined_ds = concatenate_datasets([train_ds, par_ds, syn_ds, bt_ds])

    return combined_ds

def save_tokenized_augmented_dataset(train_ds, id2label, label_names, num_syn_repl=1, num_bt=1, num_synth=3, output_path="./tokenized_augmented_dataset"):
    """
    Save the augmented dataset to disk after applying augmentation pipelines.
    """
    # Paraphrase the dataset
    augmented_ds = augment_dataset_with_paraphrasing(train_ds, id2label, label_names, num_syn_repl=num_syn_repl, num_bt=num_bt)
    syn_ds = augment_dataset_with_synonyms(train_ds, num_syn_repl=2)
    print("Paraphrasing dataset created.")
    # Tokenize the augmented dataset
    # If augmented_ds is a dict, convert it to a Dataset
    if isinstance(augmented_ds, dict):
        augmented_ds = Dataset.from_dict(augmented_ds)

    tokenized_augmented_ds = augmented_ds.map(tokenize_batch, batched=True, batch_size=1024)
    tokenized_augmented_ds.set_format("torch")
    tokenized_train_ds = train_ds.map(tokenize_batch, batched=True, batch_size=1024)
    tokenized_train_ds.set_format("torch")
    # Combine the original and augmented datasets
    combined_ds = concatenate_datasets([train_ds, augmented_ds])

    # Synonym replacements
    augmented_ds = augment_dataset_with_synonyms(train_ds, num_syn_repl=2)
    print("Synonym replacements dataset created.")
    # Tokenize the augmented dataset
    if isinstance(augmented_ds, dict):
        augmented_ds = Dataset.from_dict(augmented_ds)
    tokenized_augmented_ds = augmented_ds.map(tokenize_batch, batched=True, batch_size=1024)
    tokenized_augmented_ds.set_format("torch")
    # Combine the original and augmented datasets
    combined_ds = concatenate_datasets([combined_ds, augmented_ds])

    # Back-translation
    augmented_ds = augment_dataset_with_btranslation(train_ds, num_bt=2)
    print("Back-translation dataset created.")
    tokenized_augmented_ds = augmented_ds.map(tokenize_batch, batched=True, batch_size=1024)
    tokenized_augmented_ds.set_format("torch")
    # Combine the original and augmented datasets
    combined_ds = concatenate_datasets([combined_ds, augmented_ds])
    
    # Save the combined dataset to disk
    os.makedirs(output_path, exist_ok=True)
    print("Saving augmented dataset to file...")
    combined_ds.save_to_disk(output_path)
    return combined_ds


