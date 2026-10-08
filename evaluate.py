
import matplotlib.pyplot as plt

# Replace these examples with actual predictions
# from your trained residual-ON and residual-OFF models.

test_sentences = [
    "the cat sits on",
    "the student is",
    "we are learning",
]

results = {
    "Residual ON": {
        "validation_loss": 0.0,  # Replace with measured loss
        "probabilities": [
            [0.45, 0.30, 0.15],
            [0.40, 0.35, 0.10],
            [0.50, 0.20, 0.15],
        ],
    },
    "Residual OFF": {
        "validation_loss": 0.0,  # Replace with measured loss
        "probabilities": [
            [0.35, 0.32, 0.20],
            [0.30, 0.40, 0.15],
            [0.35, 0.30, 0.20],
        ],
    },
}

# Print a comparison table
print("VALIDATION LOSS")
for model_name, result in results.items():
    print(f"{model_name}: {result['validation_loss']}")

# Plot example prediction probabilities
# These are placeholders, not real model measurements.
for i, sentence in enumerate(test_sentences):
    plt.figure()
    positions = [0, 1, 2]
    width = 0.35

    plt.bar(
        [p - width / 2 for p in positions],
        results["Residual ON"]["probabilities"][i],
        width,
        label="Residual ON",
    )

    plt.bar(
        [p + width / 2 for p in positions],
        results["Residual OFF"]["probabilities"][i],
        width,
        label="Residual OFF",
    )

    plt.xticks(positions, ["Word 1", "Word 2", "Word 3"])
    plt.ylabel("Prediction probability")
    plt.title(f"Next-word probabilities: {sentence}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"probabilities_{i + 1}.png")
    plt.show()
