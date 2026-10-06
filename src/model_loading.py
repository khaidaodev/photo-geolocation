"""
Rebuilds the right ResNet shape for a saved model file, so the prediction and test scripts can
load either the ResNet50 or the ResNet101 models without being told which one it is.

A saved model file is just a long list of numbers, it doesn't carry the model's shape with it.
Each ResNet has a different number of blocks in its third stage (layer3): 6 in ResNet50, 23 in
ResNet101. Counting those blocks in the saved numbers tells us which empty shape to build before
loading them in.
"""

import torch
import torchvision

BUILDERS_BY_LAYER3_BLOCKS = {
    6: torchvision.models.resnet50,
    23: torchvision.models.resnet101,
}


def count_layer3_blocks(state_dict) -> int:
    """Counts how many blocks the saved numbers have in layer3, by looking at key names like
    layer3.0.conv1.weight and layer3.22.conv1.weight and counting the different block numbers."""
    block_numbers = {int(key.split(".")[1]) for key in state_dict if key.startswith("layer3.")}
    return len(block_numbers)


def build_resnet(state_dict, num_classes: int):
    """Builds an empty ResNet50 or ResNet101 (whichever matches the saved numbers) with its
    last layer sized for num_classes countries, ready to have the saved numbers loaded in."""
    blocks = count_layer3_blocks(state_dict)
    if blocks not in BUILDERS_BY_LAYER3_BLOCKS:
        raise ValueError(f"Expected a ResNet50 or ResNet101 (6 or 23 blocks in layer3), got {blocks}")
    model = BUILDERS_BY_LAYER3_BLOCKS[blocks](weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
