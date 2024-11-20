import torch

# Define transformations
transformations = [
    {"name": "Rotation", "params": {"rotation": 45}},
    {"name": "Translation", "params": {"translation": (0.2, 0.3)}},
    {"name": "Scaling", "params": {"scale": (1.5, 1.5)}},
    {"name": "Shearing", "params": {"shear": (15, 10)}},
]

# Save transformations
torch.save(transformations, "model.pth")
