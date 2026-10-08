# preprocess.py

import re
import json
import collections
from pathlib import Path

import torch


# ============================================================
# 1. FILE LOCATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Try data/sentences.txt first, then sentences.txt at repo root
_candidates = [
    PROJECT_DIR / "data" / "sentences.txt",
    PROJECT_DIR / "sentences.txt",
]
DATA_FILE = next((p for p in _candidates if p.exists()), _candidates[0])

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found. Looked in: {_candidates}"
    )


# ============================================================
# 2. SPECIAL TOKENS
# ============================================================

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"

# Must be a LIST (not a tuple) so we can concatenate with
# `vocabulary_words` using `+` later.
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


# ============================================================
# 3. SETTINGS
# ============================================================

BLOCK_SIZE = 8


# ============================================================
# 4. LOAD SENTENCES
# ============================================================

with open(DATA_FILE, "r", encoding="utf-8") as file:
    sentences = [
        line.strip()
        for line in file
        if line.strip()
    ]


# ============================================================
# 5. CLEAN TEXT
# ============================================================

def clean_sentence(text):
    """
    Clean a sentence by:
    - converting it to lowercase
    - removing unwanted characters
    - removing extra spaces
    """
    text = text.lower().strip()

    # Keep letters, spaces and apostrophes
    text = re.sub(r"[^a-z\s']", " ", text)

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Clean all sentences
sentences = [clean_sentence(s) for s in sentences]

# Remove empty sentences after cleaning
sentences = [s for s in sentences if s]


# ============================================================
# 6. TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert a sentence into a list of tokens/words.

    Example:
        "I love machine learning"
    becomes:
        ["i", "love", "machine", "learning"]
    """
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(len(sentences) * 0.9)

train_sentences = sentences[:split_index]
val_sentences = sentences[split_index:]

# Make sure validation is not empty
if len(val_sentences) == 0:
    val_sentences = train_sentences[-1:]


# ============================================================
# 8. BUILD VOCABULARY
# ============================================================

word_counts = collections.Counter()

for sentence in train_sentences:
    word_counts.update(tokenize(sentence))

# Sort words alphabetically for reproducibility
vocabulary_words = sorted(word_counts.keys())

# Add special tokens at the beginning
# SPECIAL_TOKENS is a list, so `+` works.
vocab = SPECIAL_TOKENS + vocabulary_words


# Word → number
stoi = {word: index for index, word in enumerate(vocab)}

# Number → word
itos = {index: word for word, index in stoi.items()}

# Vocabulary size
vocab_size = len(vocab)


# IDs of special tokens
PAD_ID = stoi[PAD]
UNK_ID = stoi[UNK]
BOS_ID = stoi[BOS]
EOS_ID = stoi[EOS]


# ============================================================
# 9. SAVE VOCAB FOR predict.py
# ============================================================

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)


# ============================================================
# 10. ENCODE TEXT
# ============================================================

def encode(text):
    """
    Convert words into integer token IDs.
    """
    tokens = tokenize(text)
    return [stoi.get(token, UNK_ID) for token in tokens]


# ============================================================
# 11. DECODE TOKENS
# ============================================================

def decode(ids):
    """
    Convert integer token IDs back into words.
    """
    words = []
    for token_id in ids:
        word = itos[int(token_id)]
        if word not in SPECIAL_TOKENS:
            words.append(word)
    return " ".join(words)


# ============================================================
# 12. CREATE TOKEN SEQUENCE
# ============================================================

def make_token_sequence(sentence):
    """
    Add BOS and EOS tokens to a sentence.

    Example:
        "i love ai"  →  [BOS, i, love, ai, EOS]
    """
    tokens = tokenize(sentence)
    return [BOS_ID, *[stoi.get(t, UNK_ID) for t in tokens], EOS_ID]


# ============================================================
# 13. CREATE TRAINING EXAMPLES
# ============================================================

def create_examples(sentence_list):
    """
    Create input and target sequences for next-token prediction.

    Example:
        Input:  [i, love]
        Target: [love, ai]
    """
    all_inputs = []
    all_targets = []

    for sentence in sentence_list:
        tokens = make_token_sequence(sentence)

        # We need at least two tokens
        if len(tokens) < 2:
            continue

        for i in range(1, len(tokens)):
            start = max(0, i - BLOCK_SIZE)
            input_tokens = tokens[start:i]
            target_token = tokens[i]

            # Pad input on the left if shorter than BLOCK_SIZE
            padding = [PAD_ID] * (BLOCK_SIZE - len(input_tokens))
            input_tokens = padding + input_tokens

            all_inputs.append(input_tokens)
            all_targets.append(target_token)

    return (
        torch.tensor(all_inputs, dtype=torch.long),
        torch.tensor(all_targets, dtype=torch.long),
    )


# ============================================================
# 14. CREATE TRAINING DATA
# ============================================================

X_train, y_train = create_examples(train_sentences)
X_val, y_val = create_examples(val_sentences)


# ============================================================
# 15. GET BATCH
# ============================================================

def get_batch(split="train", batch_size=32):
    """
    Return a random batch of training examples.
    split: "train" or "val"
    """
    if split == "train":
        X, y = X_train, y_train
    elif split == "val":
        X, y = X_val, y_val
    else:
        raise ValueError("split must be either 'train' or 'val'")

    if len(X) == 0:
        raise ValueError(f"No examples available for {split} split.")

    actual_batch_size = min(batch_size, len(X))
    indices = torch.randint(0, len(X), (actual_batch_size,))

    return X[indices], y[indices]


# ============================================================
# 16. INFORMATION ABOUT THE DATA
# ============================================================

print("========================================")
print("       PREPROCESSING COMPLETE")
print("========================================")

print(f"Dataset: {DATA_FILE}")
print(f"Total sentences: {len(sentences)}")
print(f"Training sentences: {len(train_sentences)}")
print(f"Validation sentences: {len(val_sentences)}")
print(f"Vocabulary size: {vocab_size}")
print(f"Training examples: {len(X_train)}")
print(f"Validation examples: {len(X_val)}")
print(f"Block size: {BLOCK_SIZE}")
print(f"Saved vocab to: {VOCAB_PATH}")

print("========================================")
print("Example")
print("========================================")

if len(sentences) > 0:
    example_sentence = sentences[0]

    print("Sentence:")
    print(example_sentence)

    print("\nTokens:")
    print(tokenize(example_sentence))

    print("\nEncoded:")
    print(encode(example_sentence))

    print("\nDecoded:")
    print(decode(encode(example_sentence)))

print("========================================")