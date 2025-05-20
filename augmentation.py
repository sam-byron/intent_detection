# augmentation.py
import random
import nltk
from nltk.corpus import wordnet
import torch
from datasets import Dataset, disable_caching, load_dataset, load_from_disk
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

def synonym_replacement(sentence: str, n: int = 6) -> str:
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

def back_translate_batch(batch, number_of_translations: int = 1):
    # English → Spanish
    en_es_tok = mt_en_es_tokenizer(batch["text"], return_tensors="pt", truncation=True, max_length=64, padding=True)
    en_es_tok = {k: v.to(device) for k, v in en_es_tok.items()}  # Move tensors to the correct device
    
    with torch.no_grad():
        en_es_translated = mt_en_es_model.generate(
            **en_es_tok, 
            num_beams=number_of_translations, 
            num_return_sequences=number_of_translations, 
            max_length=64
        )
    
    # Decode the translated text
    es_texts = mt_en_es_tokenizer.batch_decode(en_es_translated, skip_special_tokens=True)
    
    # Spanish → English
    es_en_tok = mt_es_en_tokenizer(
        es_texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=64
    )
    es_en_tok = {k: v.to(device) for k, v in es_en_tok.items()}  # Move tensors to the correct device
    
    with torch.no_grad():
        es_en_translated = mt_es_en_model.generate(
            **es_en_tok, 
            num_beams=number_of_translations, 
            num_return_sequences=number_of_translations, 
            max_length=64
        )
    
    # Decode the back-translated text
    en_texts = mt_es_en_tokenizer.batch_decode(es_en_translated, skip_special_tokens=True)
    
    # Group translations for each input
    # grouped_translations = []
    # for i in range(0, len(en_texts), number_of_translations):
    #     grouped_translations.append(en_texts[i:i + number_of_translations].join(""))
    
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
def paraphrase(text, num_return_sequences: int = 1) -> list[str]:
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

def paraphrase_batch(batch: dict, num_return_sequences: int = 1, paraph_per_seed: int = 1) -> dict:
    """
    Return a dictionary with paraphrased texts for each text in the batch.
    """
    # Prepare prompts for all texts in the batch
    prompts = [f"paraphrase: {text} </s>" for text in batch["text"] for _ in range(paraph_per_seed)]
    # outs = paraphraser(prompts, num_beams = 3, num_return_sequences=num_return_sequences, batch_size=128)
    outs = paraphraser(prompts, num_beams = 1, batch_size=128)

    # Extract the generated_text values from the output
    paraphrased_texts = [x["generated_text"].strip() for x in outs]

    # Return a dictionary with the updated "text" column
    return {"text": paraphrased_texts}

# ─── 2. Synthetic generator ───────────────────────────────────────────────────

def generate_synthetic_for_intents(
    train_ds, 
    id2label, 
    k: int = 10,
    paraph_per_seed: int = 1
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
        lambda batch: paraphrase_batch(batch, paraph_per_seed = 1),
        batched=True,
        batch_size=128
    )

    # Extract intents
    intents = [
        ex["labels"] for ex in train_ds for _ in range(paraph_per_seed)
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
    # tokenized = Dataset.from_list(tokenized['input_ids'])
    # tokenized.save_to_disk(output_paugment_dataset_with_btranslationath)

    paraphrases_ds = paraphrases_ds.rename_column("intents", "labels")
    paraphrases_ds = paraphrases_ds.rename_column("paraphrases", "text")

    # Save the combined dataset to disk
    os.makedirs(output_path, exist_ok=True)
    print("Saving augmented dataset to file...")
    paraphrases_ds.save_to_disk(output_path)
    # Convert the intent column from string labels to integer IDs

    return paraphrases_ds


def augment_dataset_with_synonyms(train_ds, num_syn_repl=5, output_path="./synonyms_aug_dataset"):
    id2label = train_ds.features["labels"].int2str
    augmented_texts = []
    augmented_labels = []
    # We assume the original batch still has a "text" field.
    print("Generating synonyms dataset...")
    for text, label in zip(train_ds["text"], train_ds["labels"]):
        # Apply synonym replacement
        synonyms_text = synonym_replacement(text, n=num_syn_repl)
        augmented_texts.append(synonyms_text)
        augmented_labels.append(label)
    syn_ds = Dataset.from_dict({"text": augmented_texts, "labels": augmented_labels})
    syn_ds.save_to_disk(output_path)
    print("Synonym replacements dataset saved.")
    # Return with the same keys as expected by the tokenizer later
    return syn_ds

def augment_dataset_with_btranslation(train_ds, id2label, num_bt=2, output_path="./backtranslation_aug_dataset"):
    # id2label = train_ds.features["intent"].int2str

    bt_ds = train_ds.map(back_translate_batch, batched=True, batch_size=128)
    # Save the back-translated dataset to disk
    os.makedirs(output_path, exist_ok=True)
    print("Saving back-translated dataset to file...")
    # Return with the same keys as expected by the tokenizer later

    # convert all values of key intent to label using id2label
    # Convert the intent column from integer IDs to string labels
    # bt_ds = bt_ds.map(lambda x: {"intent": id2label(x["intent"])}, batched=False)

    # Extract intents
    labels = [
        ex for ex in bt_ds["labels"]
    ]

    bt_ds = Dataset.from_dict({"text": bt_ds["text"], "labels": labels})
    bt_ds.save_to_disk(output_path)

    return bt_ds

def save_load_combine_tokenize_datasets():
    
    # Load the CLINC150 dataset from Hugging Face (this will download the data_full version)
    # The clinc_data object is a DatasetDict or Dataset containing all 23,700 samples in CLINC150. The dataset includes a column for the user utterance text, an intent label (e.g. "banking:balance"), a broader domain label, and a split indicator. The data is divided into training, validation, and test splits, with out-of-scope examples separated as well (e.g., "oos_train", "oos_test" for out-of-scope queries). 
    
    clinc_data = load_dataset("contemmcm/clinc150", "full")
    print(clinc_data)

    # The dataset might be stored under a single split "complete", so we filter by the 'split' field:
    if "complete" in clinc_data:
        full_dataset = clinc_data["complete"]       # All data
    else:
        full_dataset = clinc_data                  # Handle case where directly a Dataset is returned

    unique_intents = full_dataset.features["intent"].names
    id2label = full_dataset.features["intent"].int2str

    # Separate into training, validation, and test sets, including out-of-scope (oos) examples
    train_dataset = full_dataset.filter(lambda ex: ex["split"] in ["train", "oos_train"])
    val_dataset   = full_dataset.filter(lambda ex: ex["split"] in ["val", "oos_val"])
    test_dataset  = full_dataset.filter(lambda ex: ex["split"] in ["test", "oos_test"])
    # Debug
    # train_dataset = train_dataset.shuffle().select(range(1000))  # For quick testing

    # train_ds_str_intent = train_dataset.map(lambda x: {"intent": id2label(x["intent"])})  # Rename column to 'text'

    # Rename the intent column to 'labels' for compatibility with the model training API
    train_dataset = train_dataset.rename_column("intent", "labels")
    # train_dataset = train_dataset.rename_column("paraphrases", "text")

    # We can remove other columns we won't use (like 'text', 'domain', 'split') to keep dataset lean
    train_dataset = train_dataset.remove_columns(["domain", "split"])

    # Extract the ClassLabel feature from the original dataset
    label_feature = train_dataset.features["labels"]

    # Set formats for PyTorch (so that __getitem__ returns torch.Tensor)
    # train_dataset.set_format("torch")

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    # Expect ~15100 train (15000 in-scope + 100 oos), 3100 val (3000 + 100), 5500 test (4500 + 1000)

    print("Preparing augmented dataset...")

    # Get the list of intent label names (for later use in decoding predictions)
    label_names = full_dataset.features["intent"].names  # list of 151 intent labels (e.g., "banking:balance", "oos:oos", etc.)
    print(label_names)

    combined_augmented_dataset_path = "./combined_dataset"
    backtranslation_aug_dataset_path = "./backtranslation_aug_dataset"
    paraphrase_aug_dataset_path = "./paraphrase_aug_dataset"
    synonyms_aug_dataset_path = "./synonyms_aug_dataset"

    combined_ds, bt_ds, par_ds, syn_ds = [], [], [], []
    if os.path.exists(combined_augmented_dataset_path):
        print("Loading augmented dataset from file...")
        combined_ds = load_from_disk(combined_augmented_dataset_path)
        print(f"Loaded {len(combined_ds)} samples from augmented dataset.")
    else:
        if os.path.exists(backtranslation_aug_dataset_path):
            print("Loading backtranslation augmented dataset from file...")
            bt_ds = load_from_disk(backtranslation_aug_dataset_path)
            print(f"Loaded {len(bt_ds)} samples from backtranslation augmented dataset.")
        else:
            print("Augmenting dataset with back translations...")
            bt_ds = augment_dataset_with_btranslation(train_dataset, id2label)
            print("Back translation dataset created.")
        if os.path.exists(paraphrase_aug_dataset_path):
            print("Loading paraphrase augmented dataset from file...")
            par_ds = load_from_disk(paraphrase_aug_dataset_path)
            print(f"Loaded {len(par_ds)} samples from paraphrase augmented dataset.") 
        else:
            print("Augmenting dataset with paraphrasing...")
            par_ds = augment_dataset_with_paraphrasing(
                train_dataset,
                id2label,
                label_names=full_dataset.features["intent"].names,
            )
            print("Paraphrasing dataset created.")
        if os.path.exists(synonyms_aug_dataset_path):
            print("Loading synonyms augmented dataset from file...")
            syn_ds = load_from_disk(synonyms_aug_dataset_path)
            print(f"Loaded {len(syn_ds)} samples from synonyms augmented dataset.")
        else:
            syn_ds = augment_dataset_with_synonyms(train_dataset, num_syn_repl=1)

    if combined_ds == []:
         # Rename the intent column to 'labels' for compatibility with the model training API
        # par_ds = par_ds.rename_column("paraphrases", "text")
        # par_ds = par_ds.rename_column("intents", "labels")
        # bt_ds = bt_ds.rename_column("intents", "intent")
        # syn_ds = syn_ds.rename_column("labels", "intent")
        # syn_ds = syn_ds.rename_column("intent", "labels")
        # train_ds = train_ds.rename_column("intents", "intent")
        # Cast the labels column in the augmented datasets
        bt_ds = bt_ds.cast_column("labels", label_feature)
        par_ds = par_ds.cast_column("labels", label_feature)
        syn_ds = syn_ds.cast_column("labels", label_feature)
        combined_ds = concatenate_datasets([train_dataset, bt_ds, par_ds, syn_ds])
        # combined_ds = train_dataset
        # Save the combined dataset to disk
        if not os.path.exists(combined_augmented_dataset_path):
            os.makedirs(combined_augmented_dataset_path, exist_ok=True)
            print("Saved combined dataset to file...")
            combined_ds.save_to_disk(combined_augmented_dataset_path)
       
    # # DEBUG NO AUGMENTATION
    # combined_ds = train_dataset
    from transformers import AutoTokenizer

    # Initialize BERT tokenizer (uncased means text will be lowercased)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer.pad_token = tokenizer.eos_token
    #  # tell the tokenizer what its max context really is
    # tokenizer.model_max_length = config["n_positions"]

    # Tokenization function to process text examples
    def tokenize_batch(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=64)
        # max_length=64 should cover most queries; adjust if needed (max utterance length in CLINC150 is relatively short)

    # Apply tokenization to each split of the dataset
    # train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset   = val_dataset.map(tokenize_batch, batched=True)
    test_dataset  = test_dataset.map(tokenize_batch, batched=True)

    # Rename the intent column to 'labels' for compatibility with the model training API
    # train_dataset = train_dataset.rename_column("intent", "labels")
    val_dataset   = val_dataset.rename_column("intent", "labels")
    test_dataset  = test_dataset.rename_column("intent", "labels")

    # We can remove other columns we won't use (like 'text', 'domain', 'split') to keep dataset lean
    # train_dataset = train_dataset.remove_columns(["text", "domain", "split"])
    val_dataset   = val_dataset.remove_columns(["text", "domain", "split"])
    test_dataset  = test_dataset.remove_columns(["text", "domain", "split"])

    # Set formats for PyTorch (so that __getitem__ returns torch.Tensor)
    # train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Separate into training, validation, and test sets, including out-of-scope (oos) examples
    train_ds = full_dataset.filter(lambda ex: ex["split"] in ["train", "oos_train"])

    # We can remove other columns we won't use (like 'text', 'domain', 'split') to keep dataset lean
    # bt_ds = bt_ds.remove_columns(["domain", "split"])

    train_ds = train_ds.remove_columns(["domain", "split"])
    id2label = full_dataset.features["intent"].int2str
     # Extract intents
    intent = [
        id2label(ex) for ex in train_ds["intent"]
    ]
    train_ds = Dataset.from_dict({"text": train_ds["text"], "intent": intent})

    # Apply tokenization to the Combined dataset
    tokenized_train_ds_text = combined_ds.map(tokenize_batch, batched=True)
    # tokenized_train_ds_text.set_format("torch")
    tokenized_train_ds_text = tokenized_train_ds_text.map(
        lambda x: {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
    )

    # Ensure labels are integers
    # if "labels" not in augmented_train_ds.column_names:
    #     label2id = {label: i for i, label in enumerate(label_names)}
    #     augmented_train_ds = augmented_train_ds.map(
    #         lambda x: {"labels": label2id[x["intent"]]}
    #     )

    # # Extract the labels column
    # labels = augmented_train_ds["labels"]

    # TOKENIZE THE TRAINING DATASET

    # Create the dataset
    tokenized_train_ds = Dataset.from_dict({
        "input_ids": tokenized_train_ds_text["input_ids"],
        "attention_mask": tokenized_train_ds_text["attention_mask"],
        "labels": tokenized_train_ds_text["labels"],
    })
    tokenized_train_ds.set_format("torch")
    # tokenized_train_ds = tokenized_train_ds.rename_column("intent", "labels")

    # tokenize the augmented dataset
    # augmented_train = load_from_disk(augmented_dataset_path)


    # # tokenize the augmented dataset
    # tok_aug_train = augmented_train.map(tokenize_batch, batched=True)

    return combined_ds, full_dataset, train_dataset, val_dataset, test_dataset, tokenized_train_ds, tokenizer