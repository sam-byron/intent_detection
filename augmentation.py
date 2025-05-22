# augmentation.py
import random
import nltk
from nltk.corpus import wordnet
import torch
from datasets import Dataset, disable_caching, load_dataset, load_from_disk
from transformers import (
    MarianMTModel, MarianTokenizer, AutoTokenizer,
    pipeline
)
import os
from datasets import concatenate_datasets, Dataset
import multiprocessing
from multiprocessing import Pool, cpu_count
import torch.distributed as dist
from tqdm import tqdm  # Import tqdm for progress bars

device = "cuda" if torch.cuda.is_available() else "cpu"

# Disable datasets caching
# disable_caching()

# Load the CLINC150 dataset from Hugging Face (this will download the data_full version)
# The clinc_data object is a DatasetDict or Dataset containing all 23,700 samples in CLINC150. The dataset includes a column for the user utterance text, an intent label (e.g. "banking:balance"), a broader domain label, and a split indicator. The data is divided into training, validation, and test splits, with out-of-scope examples separated as well (e.g., "oos_train", "oos_test" for out-of-scope queries). 
    
clinc_data = load_dataset("contemmcm/clinc150", "full")
print(clinc_data)

# The dataset might be stored under a single split "complete", so we filter by the 'split' field:
if "complete" in clinc_data:
    full_dataset = clinc_data["complete"]       # All data
else:
    full_dataset = clinc_data                  # Handle case where directly a Dataset is returned

# full_dataset = full_dataset.remove_columns(["domain", "split"])
# Separate into training, validation, and test sets, including out-of-scope (oos) examples
train_dataset = full_dataset.filter(lambda ex: ex["split"] in ["train", "oos_train"])
train_ios_ds = full_dataset.filter(lambda ex: ex["split"] in ["train"])
val_dataset   = full_dataset.filter(lambda ex: ex["split"] in ["val", "oos_val"])
test_dataset  = full_dataset.filter(lambda ex: ex["split"] in ["test", "oos_test"])
oos_train_dataset = full_dataset.filter(lambda ex: ex["split"] in ["oos_train"])

# Rename the intent column to 'labels' for compatibility with the model training API
train_dataset = train_dataset.rename_column("intent", "labels")
train_ios_ds = train_ios_ds.rename_column("intent", "labels")
oos_train_dataset = oos_train_dataset.rename_column("intent", "labels")
val_dataset   = val_dataset.rename_column("intent", "labels")
test_dataset  = test_dataset.rename_column("intent", "labels")

# We can remove other columns we won't use (like 'text', 'domain', 'split') to keep dataset lean
test_dataset = test_dataset.remove_columns(["domain", "split"])
train_dataset = train_dataset.remove_columns(["domain", "split"])
val_dataset = val_dataset.remove_columns(["domain", "split"])
train_ios_ds = train_ios_ds.remove_columns(["domain", "split"])
oos_train_dataset = oos_train_dataset.remove_columns(["domain", "split"])

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


unique_intents = full_dataset.features["intent"].names
id2label = full_dataset.features["intent"].int2str

# ─── 0. Prepare paraphraser ───────────────────────────────────────────────────
# Use an instruction-tuned T5 model for high-quality paraphrases.
num_return_sequences=10
paraphraser = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",      # or "google/flan-t5-base" if you need smaller
    tokenizer="google/flan-t5-large",
    device=device,
    # you can tweak generation settings:
    max_length=64,
    do_sample=True, 
    top_p=0.8,
    temperature=1.4,
    repetition_penalty=1.5,
    num_return_sequences=num_return_sequences,
    top_k=100,
)

# ─── 1. Paraphrase helper ─────────────────────────────────────────────────────

def paraphrase(batch):
    # Prepare prompts for all texts in the batch
    # prompts = [f"paraphrase: {text} </s>" for text in batch["text"]]

    # Generate highly unique and relevant paraphrases capturing the intent of the original text
    prompts = [
        f'The sentence "{text}" captures the unique intent "{id2label(label)}". Generate a similar but completely unique and original sentence that captures the previous intent.'
        for text, label in zip(batch["text"], batch["labels"])
    ]

    outs = paraphraser(prompts, batch_size=64)
    # Flatten the output if necessary
    paraphrased_texts = []
    for output in outs:
        if isinstance(output, list):
            paraphrased_texts.extend([x["generated_text"] for x in output])
        else:
            paraphrased_texts.append(output["generated_text"])



     # Extract intents
    labels = [
        ex for ex in batch["labels"] for _ in range(num_return_sequences)
    ]

    # Ensure the lengths of paraphrased_texts and labels match
    assert len(paraphrased_texts) == len(labels), "Length mismatch between paraphrases and labels"

    # Return a dictionary with the updated "text" column
    return {"paraphrases": paraphrased_texts, "intents": labels}

# ─── 2. Synthetic generator ───────────────────────────────────────────────────

def generate_paraphrases(train_ds):

    all_paras_samples = {}

    all_paras_samples["paraphrases"] = []
    all_paras_samples["intents"] = []

    # paras_samples = train_ds.map(lambda d: paraphrase({'text': d['text'], 'labels': d['labels']}), batched=True, batch_size=64)

    # Process the dataset in smaller chunks manually
    paras_samples = {"paraphrases": [], "intents": []}
    for i in tqdm(range(0, len(train_ds), 64), desc="Generating paraphrases"):
        batch = train_ds[i:i + 64]  # Get a batch of 64 samples
        paraphrased_batch = paraphrase(batch)  # Generate paraphrases
        paras_samples["paraphrases"].extend(paraphrased_batch["paraphrases"])
        paras_samples["intents"].extend(paraphrased_batch["intents"])

    all_paras_samples["paraphrases"].extend(paras_samples["paraphrases"])
    all_paras_samples["intents"].extend(paras_samples["intents"])

    return all_paras_samples


# Load the tokenizer for the model we will use for tokenization
# (e.g., BERT, RoBERTa, etc.)
# This should match the model you will use for training.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(batch, padding="max_length", truncation=True, max_length=64)


def augment_dataset_with_paraphrasing(train_ds, output_path="./paraphrase_aug_dataset"):
    """
    Augment the dataset using paraphrasing, synonym translaton, and back-translation pipelines in batches.
    """
    
    # Generate synthetic texts for this intent
    # print(f"Generating synthetic text for intent: {intent}\n")
    synth_options = generate_paraphrases(train_ds)

    # augmented_ds = {"text": synth_options["paraphrases"], "labels": synth_options["intents"]}

    print("Paraphrasing dataset created.")

    paraphrases_ds = Dataset.from_dict(synth_options)
    # paraphrases_ds = synth_options
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

# ─── 1. SYNONYM REPLACEMENT ────────────────────────────────────────────────────
nltk.download('wordnet')
nltk.download('omw-1.4')

def synonym_replacement(sentence: str, iter = 10, n: int = 6) -> str:
    """
    Replace up to `n` words in `sentence` with a random WordNet synonym.
    """
    synonym_replacement_sentences = []
    # Repeat the process `n` times
    for _ in range(iter):
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
        synonym_replacement_sentences.append(" ".join(new_words))
    # Return the first generated sentence
    # (you can also return all generated sentences if needed)
    return synonym_replacement_sentences

def augment_dataset_with_synonyms(train_ds, output_path="./synonyms_aug_dataset"):
    augmented_texts = []
    augmented_labels = []
    # We assume the original batch still has a "text" field.
    print("Generating synonyms dataset...")
    for text, label in zip(train_ds["text"], train_ds["labels"]):
        # Apply synonym replacement
        synonyms_text = synonym_replacement(text)
        augmented_texts.extend(synonyms_text)
        augmented_labels.extend([label] * len(synonyms_text))
    syn_ds = Dataset.from_dict({"text": augmented_texts, "labels": augmented_labels})
    syn_ds.save_to_disk(output_path)
    print("Synonym replacements dataset saved.")
    # Return with the same keys as expected by the tokenizer later
    return syn_ds


# ─── 2. BACK-TRANSLATION ───────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use MarianMT models for translation
mt_en_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mt_en_es_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es").to(device)
mt_es_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
mt_es_en_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en").to(device)


def back_translate_batch(batch, number_of_translations: int = 10):
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

    # Extract intents
    labels = [
        ex for ex in batch["labels"] for _ in range(number_of_translations*number_of_translations)
    ]
    
    # Group translations for each input
    # grouped_translations = []
    # for i in range(0, len(en_texts), number_of_translations):
    #     grouped_translations.append(en_texts[i:i + number_of_translations].join(""))
    
    # Return a dictionary with the updated "text" column
    return {"text": en_texts, "labels": labels}

def augment_dataset_with_btranslation(train_ds, output_path="./backtranslation_aug_dataset"):
    # id2label = train_ds.features["intent"].int2str

    bt_ds = train_ds.map(back_translate_batch, batched=True, batch_size=128)
    # Save the back-translated dataset to disk
    os.makedirs(output_path, exist_ok=True)
    print("Saving back-translated dataset to file...")

    # bt_ds = Dataset.from_dict({"text": bt_ds["text"], "labels": labels})
    bt_ds = Dataset.from_dict({"text": bt_ds["text"], "labels": bt_ds["labels"]})
    bt_ds.save_to_disk(output_path)

    return bt_ds

def augment_dataset_with_oos(oos_ds, train_ds, output_path="./oos_aug_dataset"):
    # Pick all OOS examples from the original dataset
    # oos_ds = ds.filter(lambda ex: ex["split"] in ["oos_train"])
    # Paraphrase the OOS examples
    par_oos_ds = augment_dataset_with_paraphrasing(oos_ds, output_path=output_path)
    # Combine train_ds and oos_ds
    tok_oos_train_ds = {"text": [], "labels": []}
    tok_oos_train_ds["text"] = par_oos_ds["text"] + train_ds["text"]
    tok_oos_train_ds["labels"] = par_oos_ds["labels"] + train_ds["labels"]
    

    return Dataset.from_dict(tok_oos_train_ds)

def save_load_combine_tokenize_datasets():

    combined_augmented_dataset_path = "./combined_dataset"
    backtranslation_aug_dataset_path = "./backtranslation_aug_dataset"
    paraphrase_aug_dataset_path = "./paraphrase_aug_dataset"
    synonyms_aug_dataset_path = "./synonyms_aug_dataset"
    oos_aug_dataset_path = "./oos_aug_dataset"

    combined_ds, bt_ds, par_ds, syn_ds, oos_ds = [], [], [], [], []
    train_on_ds = train_ios_ds
    # train_on_ds = train_dataset
    
    if os.path.exists(backtranslation_aug_dataset_path):
        print("Loading backtranslation augmented dataset from file...")
        bt_ds = load_from_disk(backtranslation_aug_dataset_path)
        print(f"Loaded {len(bt_ds)} samples from backtranslation augmented dataset.")
    else:
        print("Augmenting dataset with back translations...")
        bt_ds = augment_dataset_with_btranslation(train_on_ds)
        print("Back translation dataset created.")
    if os.path.exists(paraphrase_aug_dataset_path):
        print("Loading paraphrase augmented dataset from file...")
        par_ds = load_from_disk(paraphrase_aug_dataset_path)
        print(f"Loaded {len(par_ds)} samples from paraphrase augmented dataset.") 
    else:
        print("Augmenting dataset with paraphrasing...")
        par_ds = augment_dataset_with_paraphrasing(
            train_on_ds,
        )
        print("Paraphrasing dataset created.")
    if os.path.exists(synonyms_aug_dataset_path):
        print("Loading synonyms augmented dataset from file...")
        syn_ds = load_from_disk(synonyms_aug_dataset_path)
        print(f"Loaded {len(syn_ds)} samples from synonyms augmented dataset.")
    else:
        print("Augmenting dataset with synonyms...")
        syn_ds = augment_dataset_with_synonyms(train_on_ds)
        print("Synonyms dataset created.")
    if os.path.exists(oos_aug_dataset_path):
        print("Loading oos augmented dataset from file...")
        oos_ds = load_from_disk(oos_aug_dataset_path)
        print(f"Loaded {len(oos_ds)} samples from oos augmented dataset.")
    else:
        print("Augmenting dataset with oos...")
        # oos_ds = augment_dataset_with_oos(oos_train_dataset, train_dataset)
        oos_ds = augment_dataset_with_oos(oos_train_dataset, combined_ds)
        print("oos dataset created.")
    
    if os.path.exists(combined_augmented_dataset_path):
        print("Loading augmented dataset from file...")
        with multiprocessing.Pool(cpu_count()-10) as pool:
            combined_ds = pool.apply(load_from_disk, args=(combined_augmented_dataset_path,))
        print(f"Loaded {len(combined_ds)} samples from augmented dataset.")
    else:
        # Cast the labels column in the augmented datasets
        bt_ds = bt_ds.cast_column("labels", label_feature)
        par_ds = par_ds.cast_column("labels", label_feature)
        syn_ds = syn_ds.cast_column("labels", label_feature)
        combined_ds = concatenate_datasets([train_on_ds, bt_ds, par_ds, syn_ds])
        # combined_ds = concatenate_datasets([train_dataset, par_ds, syn_ds])
        # combined_ds = train_dataset
        # Save the combined dataset to disk
        if not os.path.exists(combined_augmented_dataset_path):
            os.makedirs(combined_augmented_dataset_path, exist_ok=True)
            print("Saved combined dataset to file...")
            combined_ds.save_to_disk(combined_augmented_dataset_path)


    
    return combined_ds, full_dataset, train_dataset, val_dataset, test_dataset, oos_ds, tokenizer