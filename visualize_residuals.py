import matplotlib.pyplot as plt
import torch


def visualize_residual(
    input_representation,
    attention_output,
    residual_on=True
):
    """
    Visualize the effect of the residual connection.

    Residual ON:
        combined = input_representation + attention_output

    Residual OFF:
        combined = attention_output
    """

    # Make sure tensors do not require gradients
    input_representation = input_representation.detach().cpu()
    attention_output = attention_output.detach().cpu()

    if residual_on:
        combined = input_representation + attention_output
        title = "Residual Connection ON"
    else:
        combined = attention_output
        title = "Residual Connection OFF"

    # Use the first sample and first token
    input_vector = input_representation[0, 0]
    attention_vector = attention_output[0, 0]
    combined_vector = combined[0, 0]

    # Convert to Python/NumPy values for plotting
    input_vector = input_vector.numpy()
    attention_vector = attention_vector.numpy()
    combined_vector = combined_vector.numpy()

    # Create the plot
    plt.figure(figsize=(12, 5))

    plt.plot(input_vector, label="Input representation")
    plt.plot(attention_vector, label="Attention output")
    plt.plot(combined_vector, label="After residual operation")

    plt.title(title)
    plt.xlabel("Vector dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.show()


def compare_residuals(
    input_representation,
    attention_output
):
    """
    Compare Residual ON and Residual OFF.
    """

    input_representation = input_representation.detach().cpu()
    attention_output = attention_output.detach().cpu()

    # Residual ON
    residual_on = input_representation + attention_output

    # Residual OFF
    residual_off = attention_output

    # First sample, first token
    input_vector = input_representation[0, 0].numpy()
    attention_vector = attention_output[0, 0].numpy()
    residual_on_vector = residual_on[0, 0].numpy()
    residual_off_vector = residual_off[0, 0].numpy()

    # Plot
    plt.figure(figsize=(12, 6))

    plt.plot(input_vector, label="Input representation")
    plt.plot(attention_vector, label="Attention output")
    plt.plot(
        residual_on_vector,
        label="Residual ON: x + attention"
    )
    plt.plot(
        residual_off_vector,
        label="Residual OFF: attention only"
    )

    plt.title("Residual Connection Comparison")
    plt.xlabel("Vector dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.show()


def print_residual_values(
    input_representation,
    attention_output
):
    """
    Print a small number of actual tensor values.
    Useful for explaining the residual connection
    during the presentation.
    """

    input_representation = input_representation.detach().cpu()
    attention_output = attention_output.detach().cpu()

    residual_on = input_representation + attention_output
    residual_off = attention_output

    print("Input representation:")
    print(input_representation[0, 0, :10])

    print("\nAttention output:")
    print(attention_output[0, 0, :10])

    print("\nResidual ON (input + attention):")
    print(residual_on[0, 0, :10])

    print("\nResidual OFF (attention only):")
    print(residual_off[0, 0, :10])
