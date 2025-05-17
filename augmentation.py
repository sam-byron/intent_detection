# augmentation.py
import random
import nltk
from nltk.corpus import wordnet
import torch
from datasets import Dataset, disable_caching
from transformers import (
    MarianMTModel, MarianTokenizer, AutoTokenizer,
    pipeline as hf_pipeline,
)
import os
from datasets import concatenate_datasets
import multiprocessing
from multiprocessing import Pool, cpu_count
import torch.distributed as dist

# Disable datasets caching
disable_caching()

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

def back_translate(text: str) -> str:
    # English → Spanish
    batch = mt_en_es_tokenizer([text], return_tensors="pt", truncation=True, max_length=64)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        translated = mt_en_es_model.generate(**batch, num_beams=1, max_length=64)
    es = mt_en_es_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    # Spanish → English
    batch_rev = mt_es_en_tokenizer([es], return_tensors="pt", truncation=True, max_length=64)
    batch_rev = {k: v.to(device) for k, v in batch_rev.items()}
    with torch.no_grad():
        back = mt_es_en_model.generate(**batch_rev, num_beams=1, max_length=64)
    return mt_es_en_tokenizer.batch_decode(back, skip_special_tokens=True)[0]

# ─── 3. PROMPT-BASED SYNTHETIC GENERATION ───────────────────────────────────────
# Here we use a small GPT-style model for demonstration; swap in a larger LLM if you like.
gen_pipe = hf_pipeline(
    "text-generation",
    model="gpt2",
    tokenizer="gpt2",
    device=0  # or -1 for CPU
)

def generate_synthetic_for_intent(intent_label: str, k: int = 5) -> list[str]:
    """
    Use a text-generation model to produce k example utterances for a given intent.
    """
    prompt = (
        f"Generate {k} different ways a user might ask for the intent “{intent_label}”:\n"
        "1."
    )
    outputs = gen_pipe(
        prompt,
        max_length=64,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1  # we embed all k in one sequence
    )
    text = outputs[0]["generated_text"]
    # parse lines after the “1.” prompt into a list
    lines = text.splitlines()
    # lines like ["1. how do i check my balance?", "2. what's my bank balance?", ...]
    examples = []
    for line in lines:
        # strip leading numbering
        parts = line.strip().split(".", 1)
        if len(parts) == 2 and parts[0].isdigit():
            examples.append(parts[1].strip())
    return examples[:k]


# Load the tokenizer for the model we will use for tokenization
# (e.g., BERT, RoBERTa, etc.)
# This should match the model you will use for training.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def augment_dataset(batch, num_syn_repl=1, num_bt=1):
    augmented_texts = []
    augmented_labels = []
    # We assume the original batch still has a "text" field.
    for text, label in zip(batch["text"], batch["labels"]):
        # Apply synonym replacement
        aug_text = synonym_replacement(text, n=num_syn_repl)
        # Optionally apply back-translation (uncomment the next two lines if desired)
        bt_text = back_translate(text)
        # aug_text = bt_text  # choose which augmentation to keep
        augmented_texts.append(aug_text)
        augmented_labels.append(label)
        augmented_texts.append(bt_text)
        augmented_labels.append(label)
    # Return with the same keys as expected by the tokenizer later
    return {"text": augmented_texts, "labels": augmented_labels}

def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

def save_augmented_dataset(train_ds, num_syn_repl=1, num_bt=1, num_synth=3, output_path="./augmented_dataset"):
    # Here we perform augmentation by mapping over the training dataset
    augmented_ds = train_ds.map(
        lambda batch: augment_dataset(batch, num_syn_repl=num_syn_repl, num_bt=num_bt),
        batched=True,
        batch_size=512,  # Specify the batch size here
        remove_columns=train_ds.column_names  # remove old columns if necessary
    )
    # Now tokenize the augmented text so that we get 'input_ids', 'attention_mask', etc.
    augmented_ds = augmented_ds.map(tokenize_batch, batched=True, batch_size=512)
    augmented_ds.set_format("torch")
     # Concatenate augmented_ds with the original dataset.
    combined_ds = concatenate_datasets([train_ds, augmented_ds])
    
    # Save the augmented dataset to disk
    os.makedirs(output_path, exist_ok=True)
    print("Saving augmented dataset to file...")
    combined_ds.save_to_disk(output_path)
    return combined_ds
   

if __name__ == '__main__':
    dist.init_process_group(backend="nccl")
    multiprocessing.set_start_method("spawn", force=True)

