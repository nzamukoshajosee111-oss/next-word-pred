import json
from pathlib import Path

import torch
from model import TinyWordGPT

# Import the exact block size used during training
from preprocess import BLOCK_SIZE


# --------------------------------------------------
# 1. File paths and device
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"

RESIDUAL_ON_PATH = ARTIFACTS_DIR / "model_residual_on.pt"
RESIDUAL_OFF_PATH = ARTIFACTS_DIR / "model_residual_off.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Load vocabulary
# --------------------------------------------------

def load_vocabulary():
    """Load the word-to-ID vocabulary saved during training."""

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {VOCAB_PATH}"
        )

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        raw = json.load(file)

    # Support two formats:
    #   1) flat dict:           {"word": id, ...}
    #   2) {"vocab": [...], ...}
    if isinstance(raw, dict) and "vocab" in raw and isinstance(raw["vocab"], list):
        words = raw["vocab"]
        vocab = {word: idx for idx, word in enumerate(words)}
    elif isinstance(raw, dict):
        vocab = raw
    else:
        raise ValueError("vocab.json has an unsupported structure.")

    try:
        vocab = {str(w): int(i) for w, i in vocab.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("Vocabulary IDs must be integers.") from error

    if not vocab:
        raise ValueError("Vocabulary is empty.")

    return vocab


# --------------------------------------------------
# 3. Load the trained Transformer model
# --------------------------------------------------

def load_model(use_residual=True):
    """
    Load the appropriate model checkpoint.

    use_residual=True  -> residual connections ON
    use_residual=False -> residual connections OFF
    """

    checkpoint_path = (
        RESIDUAL_ON_PATH if use_residual else RESIDUAL_OFF_PATH
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    vocab = load_vocabulary()

    # Must match train.py exactly — same block_size!
    model = TinyWordGPT(
        vocab_size=len(vocab),
        block_size=BLOCK_SIZE,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        use_residual=use_residual,
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, vocab


# --------------------------------------------------
# 4. Convert input text into token IDs
# --------------------------------------------------

def encode_text(text, vocab):
    """
    Convert a sentence into token IDs.
    Must match preprocess.tokenize() behavior.
    """
    import re
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

    unknown_id = vocab.get("<UNK>")

    token_ids = []
    for word in words:
        if word in vocab:
            token_ids.append(vocab[word])
        elif unknown_id is not None:
            token_ids.append(unknown_id)

    return token_ids


# --------------------------------------------------
# 5. Predict the next words
# --------------------------------------------------

def suggest_next_words(text, top_k=3, use_residual=True):
    """
    Return a list of dicts:
    [{"word": "happy", "probability": 0.42}, ...]
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if not isinstance(use_residual, bool):
        raise TypeError("use_residual must be True or False.")

    if not text.strip():
        return []

    model, vocab = load_model(use_residual=use_residual)

    token_ids = encode_text(text, vocab)

    # If the sentence has no known tokens, return nothing.
    if not token_ids:
        return []

    # Truncate to the last BLOCK_SIZE tokens (model can only see that far)
    token_ids = token_ids[-BLOCK_SIZE:]

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():
        output = model(input_tensor)

        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, dict):
            if "logits" not in output:
                raise ValueError("Model output dict must contain 'logits'.")
            output = output["logits"]

        if output.ndim == 3:
            logits = output[0, -1, :]
        elif output.ndim == 2:
            logits = output[0]
        else:
            raise ValueError(
                "Unexpected model output shape. "
                "Expected [batch, seq, vocab] or [batch, vocab]."
            )

        probabilities = torch.softmax(logits, dim=-1)

        number_to_return = min(top_k, probabilities.numel())

        top_probabilities, top_ids = torch.topk(
            probabilities,
            number_to_return,
        )

    id_to_word = {index: word for word, index in vocab.items()}

    special_tokens = {
        "<pad>", "<unk>", "<bos>", "<eos>",
        "<PAD>", "<UNK>", "<BOS>", "<EOS>",
    }

    suggestions = []

    for token_id, probability in zip(
        top_ids.tolist(),
        top_probabilities.tolist(),
    ):
        word = id_to_word.get(token_id)

        if word is None or word in special_tokens:
            continue

        suggestions.append({
            "word": word,
            "probability": float(probability),
        })

    return suggestions