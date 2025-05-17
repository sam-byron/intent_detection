# augmentation.py
import random
import nltk
from nltk.corpus import wordnet
from datasets import Dataset
from transformers import (
    MarianMTModel, MarianTokenizer,
    pipeline as hf_pipeline,
    AutoTokenizer
)

# ─── 1. SYNONYM REPLACEMENT ────────────────────────────────────────────────────
nltk.download('wordnet')
nltk.download('omw-1.4')

def synonym_replacement(sentence: str, n: int = 2) -> str:
    """
    Replace up to `n` words in `sentence` with a random WordNet synonym.
    """
    words = sentence.split()
    candidates = [w for w in words if wordnet.synsets(w)]
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
# Load English→Spanish and Spanish→English models once
mt_en_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mt_en_es_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mt_es_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
mt_es_en_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

def back_translate(text: str) -> str:
    # English → Spanish
    batch = mt_en_es_tokenizer([text], return_tensors="pt", truncation=True, max_length=128)
    translated = mt_en_es_model.generate(**batch, num_beams=5, max_length=128)
    es = mt_en_es_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    # Spanish → English
    batch_rev = mt_es_en_tokenizer([es], return_tensors="pt", truncation=True, max_length=128)
    back = mt_es_en_model.generate(**batch_rev, num_beams=5, max_length=128)
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

def augment_dataset(train_ds, num_syn_repl=1, num_bt=1, num_synth=3):
    """
    Given a HuggingFace Dataset `train_ds` with fields ["text","labels","input_ids",...],
    return an augmented Dataset.
    """
    # 1) Collect original examples
    print("Collecting original examples...")
    orig_dict = {k: train_ds[k] for k in train_ds.column_names}
    print(f"Original dataset columns: {list(orig_dict.keys())}")
    print(f"Number of original examples: {len(orig_dict['text'])}")

    # 2) Build lists for augmented samples
    aug_texts  = []
    aug_labels = []

    for text, label in zip(orig_dict["text"], orig_dict["labels"]):
        print(f"Processing text: {text}, label: {label}")
        # synonym replacements
        for _ in range(num_syn_repl):
            augmented_text = synonym_replacement(text)
            print(f"Synonym replacement: {augmented_text}")
            aug_texts.append(augmented_text)
            aug_labels.append(label)
        # back-translations
        # for _ in range(num_bt):
        #     augmented_text = back_translate(text)
        #     print(f"Back-translated text: {augmented_text}")
        #     aug_texts.append(augmented_text)
        #     aug_labels.append(label)

    # 3) Synthetic per-intent generation
    # print("Generating synthetic examples per intent...")
    # unique_labels = set(aug_labels)
    # print(f"Unique labels: {unique_labels}")
    # for lbl in unique_labels:
    #     intent_name = train_ds.features["labels"].int2str(lbl)
    #     print(f"Generating synthetic examples for intent: {intent_name}")
    #     synths = generate_synthetic_for_intent(intent_name, k=num_synth)
    #     for s in synths:
    #         print(f"Synthetic example: {s}")
    #         aug_texts.append(s)
    #         aug_labels.append(lbl)

    # 4) Create a new Dataset
    print("Creating augmented dataset...")
    aug_ds = Dataset.from_dict({"text": aug_texts, "labels": aug_labels})

    # Align the feature type of the 'labels' column with the original dataset
    aug_ds = aug_ds.cast_column("labels", train_ds.features["labels"])
    print(f"Number of augmented examples: {len(aug_ds)}")

    # 5) Concatenate & re-tokenize
    from datasets import concatenate_datasets
    print("Concatenating original and augmented datasets...")
    combined = concatenate_datasets([train_ds, aug_ds])
    print(f"Total number of examples after concatenation: {len(combined)}")

    print("Re-tokenizing combined dataset...")
    combined = combined.map(
        lambda ex: tokenizer(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=64
        ),
        batched=True
    )

    # Drop old columns if necessary
    print("Removing unnecessary columns...")
    combined = combined.remove_columns(["text"])  # keep only input_ids, attention_mask, labels
    combined.set_format("torch")
    print("Augmentation complete. Returning augmented dataset.")
    return combined

