"""
Generate LAMB optimizer test data using TIMM's Lamb implementation.

Architecture matches the C++ test: MLP with INPUT_DIM=4, HIDDEN_DIM=32,
OUTPUT_DIM=4, 2 layers (input + output), ReLU hidden, identity output.

TIMM Lamb config:
  - always_adapt=True enables trust ratio even when weight_decay=0
  - max_grad_norm=None disables global gradient clipping
  - trust_clip=False means no upper clamp on trust ratio
  - grad_averaging=True (default) uses standard (1-beta1) scaling
  - bias_correction=True applies Adam-style bias correction

Usage:
    pip install timm h5py torch
    python generate_test_data.py
"""

import os
import h5py
import torch
import torch.nn as nn

try:
    from timm.optim import Lamb
except ImportError:
    raise ImportError("timm is required: pip install timm")

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

INPUT_DIM = 4
HIDDEN_DIM = 32
OUTPUT_DIM = 4
N_STEPS = 10

LR = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-7
WEIGHT_DECAY = 0.0


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.output_layer = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        return self.output_layer(x)


def save_layer(group, name, layer):
    lg = group.require_group(name)
    lg.create_dataset("weight", data=layer.weight.detach().numpy())
    lg.create_dataset("bias", data=layer.bias.detach().numpy())


def save_layer_grads(group, name, layer):
    lg = group.require_group(name)
    lg.create_dataset("weight", data=layer.weight.grad.detach().numpy())
    lg.create_dataset("bias", data=layer.bias.grad.detach().numpy())


def save_optimizer_state(group, model, optimizer):
    for name, param in model.named_parameters():
        state = optimizer.state[param]
        layer_name, param_name = name.split(".")
        lg = group.require_group(layer_name)
        pg = lg.create_group(param_name)
        pg.create_dataset("exp_avg", data=state["exp_avg"].numpy())
        pg.create_dataset("exp_avg_sq", data=state["exp_avg_sq"].numpy())
    group.attrs["step"] = optimizer.param_groups[0]["step"]


def main():
    model = MLP()
    optimizer = Lamb(
        model.parameters(),
        lr=LR,
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=None,
        trust_clip=False,
        always_adapt=True,
        bias_correction=True,
        grad_averaging=True,
    )

    input_data = torch.randn(1, INPUT_DIM)
    target_data = torch.randn(1, OUTPUT_DIM)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(
        os.path.join(script_dir, "..", "..", "..", "..", "..")
    )
    output_dir = os.path.join(repo_root, "tests", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lamb_test_data.h5")

    print("Model: Linear(%d,%d) -> ReLU -> Linear(%d,%d)"
          % (INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM))
    print("LAMB: lr=%s, beta1=%s, beta2=%s, eps=%s, wd=%s"
          % (LR, BETA1, BETA2, EPS, WEIGHT_DECAY))
    print()

    with h5py.File(output_path, "w") as f:
        config = f.create_group("config")
        config.attrs["input_dim"] = INPUT_DIM
        config.attrs["hidden_dim"] = HIDDEN_DIM
        config.attrs["output_dim"] = OUTPUT_DIM
        config.attrs["n_steps"] = N_STEPS
        config.attrs["lr"] = LR
        config.attrs["beta1"] = BETA1
        config.attrs["beta2"] = BETA2
        config.attrs["eps"] = EPS
        config.attrs["weight_decay"] = WEIGHT_DECAY

        f.create_dataset("input", data=input_data.numpy())
        f.create_dataset("target", data=target_data.numpy())

        init_group = f.create_group("init")
        save_layer(init_group, "input_layer", model.input_layer)
        save_layer(init_group, "output_layer", model.output_layer)

        steps_group = f.create_group("steps")
        for step_i in range(N_STEPS):
            sg = steps_group.create_group(str(step_i))

            optimizer.zero_grad()
            output = model(input_data)
            loss = nn.MSELoss()(output, target_data)
            loss.backward()

            sg.create_dataset("output", data=output.detach().numpy())
            sg.attrs["loss"] = loss.item()

            d_loss = 2.0 * (output.detach() - target_data) / OUTPUT_DIM
            sg.create_dataset("d_loss_d_output", data=d_loss.numpy())

            gg = sg.create_group("gradients")
            save_layer_grads(gg, "input_layer", model.input_layer)
            save_layer_grads(gg, "output_layer", model.output_layer)

            optimizer.step()

            wg = sg.create_group("weights")
            save_layer(wg, "input_layer", model.input_layer)
            save_layer(wg, "output_layer", model.output_layer)

            og = sg.create_group("optimizer_state")
            save_optimizer_state(og, model, optimizer)

            print("Step %d: loss=%.10f" % (step_i, loss.item()))

    print("\nSaved to " + output_path)


if __name__ == "__main__":
    main()
