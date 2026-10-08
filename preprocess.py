
import re
import collections
from pathlib import Path

import torch


# ============================================================
# 1. FILE LOCATION
# ============================================================

# Find the project folder where this file is located
PROJECT_DIR = Path(__file__).resolve().parent

# Dataset location
DATA_FILE = PROJECT_DIR / "data" / "sentences.txt"


# ============================================================
# 2. SPECIAL TOKENS
# ============================================================

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


# ============================================================
# 3. SETTINGS
# ============================================================

# Number of previous tokens the model looks at
BLOCK_SIZE = 8


# ============================================================
# 4. LOAD SENTENCES
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_FILE}\n"
        "Make sure you have data/sentences.txt in your project."
    )


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
sentences = [
    clean_sentence(sentence)
    for sentence in sentences
]


# Remove empty sentences after cleaning
sentences = [
    sentence
    for sentence in sentences
    if sentence
]


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

    return re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        text.lower()
    )


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
vocab = SPECIAL_TOKENS + vocabulary_words


# Word → number
stoi = {
    word: index
    for index, word in enumerate(vocab)
}


# Number → word
itos = {
    index: word
    for word, index in stoi.items()
}


# Vocabulary size
vocab_size = len(vocab)


# IDs of special tokens
PAD_ID = stoi[PAD]
UNK_ID = stoi[UNK]
BOS_ID = stoi[BOS]
EOS_ID = stoi[EOS]


# ============================================================
# 9. ENCODE TEXT
# ============================================================

def encode(text):
    """
    Convert words into integer token IDs.

    Example:

        "i love ai"

    might become:

        [4, 7, 12]
    """

    tokens = tokenize(text)

    return [
        stoi.get(token, UNK_ID)
        for token in tokens
    ]


# ============================================================
# 10. DECODE TOKENS
# ============================================================

def decode(ids):
    """
    Convert integer token IDs back into words.
    """

    words = []

    for token_id in ids:

        word = itos[int(token_id)]

        # Don't display special tokens
        if word not in SPECIAL_TOKENS:
            words.append(word)

    return " ".join(words)


# ============================================================
# 11. CREATE TOKEN SEQUENCE
# ============================================================

def make_token_sequence(sentence):
    """
    Add BOS and EOS tokens to a sentence.

    Example:

        "i love ai"

    becomes:

        [BOS, i, love, ai, EOS]
    """

    tokens = tokenize(sentence)

    return [
        BOS_ID,
        *[
            stoi.get(token, UNK_ID)
            for token in tokens
        ],
        EOS_ID
    ]


# ============================================================
# 12. CREATE TRAINING EXAMPLES
# ============================================================

def create_examples(sentence_list):
    """
    Create input and target sequences for next-token prediction.

    Example:

        Input:  [i, love]
        Target: [love, ai]

    The model learns to predict the next token.
    """

    all_inputs = []
    all_targets = []

    for sentence in sentence_list:

        tokens = make_token_sequence(sentence)

        # We need at least two tokens
        if len(tokens) < 2:
            continue

        # Create next-token prediction examples
        for i in range(1, len(tokens)):

            start = max(0, i - BLOCK_SIZE)

            input_tokens = tokens[start:i]
            target_token = tokens[i]

            # Pad input on the left if it is shorter
            # than BLOCK_SIZE
            padding = [PAD_ID] * (
                BLOCK_SIZE - len(input_tokens)
            )

            input_tokens = padding + input_tokens

            all_inputs.append(input_tokens)
            all_targets.append(target_token)

    return (
        torch.tensor(all_inputs, dtype=torch.long),
        torch.tensor(all_targets, dtype=torch.long)
    )


# ============================================================
# 13. CREATE TRAINING DATA
# ============================================================

X_train, y_train = create_examples(train_sentences)

X_val, y_val = create_examples(val_sentences)


# ============================================================
# 14. GET BATCH
# ============================================================

def get_batch(split="train", batch_size=32):
    """
    Return a random batch of training examples.

    split:
        "train" → training data
        "val"   → validation data
    """

    if split == "train":

        X = X_train
        y = y_train

    elif split == "val":

        X = X_val
        y = y_val

    else:

        raise ValueError(
            "split must be either 'train' or 'val'"
        )

    # Make sure data exists
    if len(X) == 0:
        raise ValueError(
            f"No examples available for {split} split."
        )

    # Number of examples to sample
    actual_batch_size = min(
        batch_size,
        len(X)
    )

    # Random indices
    indices = torch.randint(
        0,
        len(X),
        (actual_batch_size,)
    )

    return X[indices], y[indices]


# ============================================================
# 15. INFORMATION ABOUT THE DATA
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

