
import torch
import numpy as np

from model import TinyWordGPT
from preprocess import (
    vocab_size,
    BLOCK_SIZE,
    get_batch,
)


# ============================================================
# 1. DEVICE
# ============================================================

# Use GPU if one is available.
# Otherwise, use the CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2

LEARNING_RATE = 0.001
TRAINING_STEPS = 600
BATCH_SIZE = 32


# ============================================================
# 3. CREATE MODEL
# ============================================================

model = TinyWordGPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    use_residual=True,
)

model = model.to(device)


print("\n========================================")
print("MODEL CREATED")
print("========================================")
print(model)


# ============================================================
# 4. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 5. EVALUATE LOSS
# ============================================================

def evaluate_loss(split="val", batches=20):
    """
    Calculate the average loss on either the
    training or validation data.
    """

    model.eval()

    losses = []

    with torch.no_grad():

        for _ in range(batches):

            x, y = get_batch(
                split,
                batch_size=BATCH_SIZE
            )

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            losses.append(loss.item())

    model.train()

    return float(np.mean(losses))


# ============================================================
# 6. TRAINING FUNCTION
# ============================================================

def train_model(steps=TRAINING_STEPS):

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
    }

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================")

    for step in range(1, steps + 1):

        # ----------------------------------------------------
        # Get a random batch
        # ----------------------------------------------------

        x, y = get_batch(
            "train",
            batch_size=BATCH_SIZE
        )

        x = x.to(device)
        y = y.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits, loss = model(x, y)

        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Prevent extremely large gradients
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # ----------------------------------------------------
        # Update model parameters
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        if step == 1 or step % 50 == 0:

            train_loss = evaluate_loss(
                "train"
            )

            val_loss = evaluate_loss(
                "val"
            )

            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Step {step:4d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Validation Loss: {val_loss:.4f}"
            )

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")

