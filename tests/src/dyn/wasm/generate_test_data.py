#!/usr/bin/env python3
"""Generate test HDF5 checkpoint with model weights, test input, and expected output."""
import numpy as np
import h5py
import sys

def relu(x):
    return np.maximum(x, 0)

def dense_forward(x, weights, biases, activation):
    """Forward pass for a dense layer. weights: [output_dim, input_dim], biases: [output_dim, 1]"""
    out = x @ weights.T + biases.reshape(1, -1)
    if activation == "RELU":
        out = relu(out)
    return out

def save_dense(group, weights, biases, activation):
    group.attrs["type"] = "dense"
    group.attrs["activation_function"] = activation
    wg = group.create_group("weights")
    wg.create_dataset("parameters", data=weights)
    bg = group.create_group("biases")
    bg.create_dataset("parameters", data=biases)

def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "test_checkpoint.h5"
    np.random.seed(42)

    input_dim = 8
    hidden_dim = 16
    output_dim = 4
    batch_size = 3

    # MLP: input_layer(RELU) -> hidden_layer_0(RELU) -> output_layer(IDENTITY)
    w_input = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.5
    b_input = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.1
    w_hidden = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.5
    b_hidden = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.1
    w_output = np.random.randn(output_dim, hidden_dim).astype(np.float32) * 0.5
    b_output = np.random.randn(output_dim, 1).astype(np.float32) * 0.1

    # Test input
    test_input = np.random.randn(batch_size, input_dim).astype(np.float32)

    # Forward pass
    h1 = dense_forward(test_input, w_input, b_input, "RELU")
    h2 = dense_forward(h1, w_hidden, b_hidden, "RELU")
    expected_output = dense_forward(h2, w_output, b_output, "IDENTITY")

    with h5py.File(output_path, "w") as f:
        # Save MLP model
        model = f.create_group("model")
        model.attrs["type"] = "mlp"
        model.attrs["num_layers"] = "3"  # input + 1 hidden + output

        save_dense(model.create_group("input_layer"), w_input, b_input, "RELU")
        hidden_layers = model.create_group("hidden_layers")
        save_dense(hidden_layers.create_group("0"), w_hidden, b_hidden, "RELU")
        save_dense(model.create_group("output_layer"), w_output, b_output, "IDENTITY")

        # Save test data (flat)
        f.create_dataset("test_input", data=test_input.flatten())
        f.attrs["input_dim"] = str(input_dim)
        f.attrs["batch_size"] = str(batch_size)
        f.create_dataset("expected_output", data=expected_output.flatten())

    print(f"Wrote {output_path}")
    print(f"  Model: MLP {input_dim} -> {hidden_dim}(RELU) -> {hidden_dim}(RELU) -> {output_dim}(IDENTITY)")
    print(f"  Test input: [{batch_size}, {input_dim}]")
    print(f"  Expected output: [{batch_size}, {output_dim}]")
    print(f"  Output sample: {expected_output[0]}")

if __name__ == "__main__":
    main()
