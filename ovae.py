# ovae.py
import torch
import torch.nn as nn
import json
import numpy as np


class OrganicEncoder(nn.Module):
    """
    O-VAE encoder class.

    IMPORTANT:
    - Replace the architecture in __init__ with the exact one you used when
      you generated encoder_weights.json.
    - The JSON is assumed to be a dict mapping parameter names -> nested lists
      that can be turned into tensors.

    By default, this will try to load "encoder_weights.json" from the same
    directory on CPU.
    """

    def __init__(self, weights_path: str = "encoder_weights.json",
                 device: str = "cpu",
                 load_weights: bool = True):
        super().__init__()

        # ---------------------------------------------------------
        # TODO: REPLACE THIS WITH YOUR REAL O-VAE ARCHITECTURE
        # This is JUST a placeholder to show the wiring.
        # It must match the shapes stored in encoder_weights.json.
        # ---------------------------------------------------------
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 4),
        )
        # ---------------------------------------------------------

        if load_weights:
            self.load_from_json(weights_path, device=device)

    def load_from_json(self, json_path: str, device: str = "cpu"):
        """
        Load weights from a JSON file.

        Expected JSON format (example):
        {
            "net.0.weight": [[[[...], ...], ...]],
            "net.0.bias": [...],
            ...
        }

        The keys must match the names in model.state_dict().
        """
        # Move to device before or after loading; both are fine
        self.to(device)

        with open(json_path, "r") as f:
            raw = json.load(f)

        state_dict = {}
        for key, value in raw.items():
            # Convert nested lists -> numpy -> tensor
            arr = np.array(value, dtype=np.float32)
            tensor = torch.from_numpy(arr)
            state_dict[key] = tensor

        # strict=False lets you partially load if JSON does not include
        # every key, but ideally it matches exactly.
        self.load_state_dict(state_dict, strict=False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
