


import json
from pathlib import Path

import torch
from model import TransformerModel


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
        vocab = json.load(file)

    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(
            "vocab.json must contain a non-empty word-to-ID dictionary."
        )

    # Validate that vocabulary IDs are integers.
    try:
        vocab = {word: int(index) for word, index in vocab.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Vocabulary IDs must be integers."
        ) from error

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
        RESIDUAL_ON_PATH
        if use_residual
        else RESIDUAL_OFF_PATH
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    vocab = load_vocabulary()

    # This constructor must match Member 1's model.py.
    model = TransformerModel(
        vocab_size=len(vocab),
        use_residual=use_residual
    )

    # Checkpoints should contain a state_dict or a dictionary
    # with a "model_state_dict" entry.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
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

    This simple version splits on whitespace.
    Match this behavior to preprocess.py.
    """

    words = text.lower().strip().split()

    unknown_id = vocab.get("<unk>")

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
    Return a list of dictionaries containing suggested words
    and their probabilities.

    Example:
    [
        {"word": "happy", "probability": 0.42},
        {"word": "ready", "probability": 0.25},
        {"word": "here", "probability": 0.12}
    ]
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer.")

    if not isinstance(use_residual, bool):
        raise TypeError("use_residual must be True or False.")

    if not text.strip():
        return []

    # Load the appropriate model and vocabulary.
    model, vocab = load_model(use_residual=use_residual)

    # Convert the sentence into token IDs.
    token_ids = encode_text(text, vocab)

    if not token_ids:
        return []

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    # Run inference without calculating gradients.
    with torch.inference_mode():
        output = model(input_tensor)

        # Support models that return (logits, other_values).
        if isinstance(output, tuple):
            output = output[0]

        # Support models that return {"logits": ...}.
        elif isinstance(output, dict):
            if "logits" not in output:
                raise ValueError(
                    "Model output dictionary must contain 'logits'."
                )
            output = output["logits"]

        # Expected output:
        # [batch_size, sequence_length, vocabulary_size]
        # or [batch_size, vocabulary_size]
        if output.ndim == 3:
            logits = output[0, -1, :]

        elif output.ndim == 2:
            logits = output[0]

        else:
            raise ValueError(
                "Unexpected model output shape. "
                "Expected [batch, sequence, vocab] "
                "or [batch, vocab]."
            )

        probabilities = torch.softmax(logits, dim=-1)

        # Limit the number of suggestions to the vocabulary size.
        number_to_return = min(
            top_k,
            probabilities.numel()
        )

        top_probabilities, top_ids = torch.topk(
            probabilities,
            number_to_return
        )

    # Convert vocabulary IDs back into words.
    id_to_word = {
        index: word
        for word, index in vocab.items()
    }

    special_tokens = {
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>"
    }

    suggestions = []

    for token_id, probability in zip(
        top_ids.tolist(),
        top_probabilities.tolist()
    ):
        word = id_to_word.get(token_id)

        if word is None or word in special_tokens:
            continue

        suggestions.append({
            "word": word,
            "probability": float(probability)
        })

    return suggestions
