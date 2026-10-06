"""
Tests for model_loading.py, using empty ResNets built from scratch instead of real trained models.
"""

import sys
from pathlib import Path

import pytest
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import model_loading


def _saved_numbers_for(builder, num_classes=3):
    model = builder(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model.state_dict()


def test_count_layer3_blocks_tells_resnet50_and_resnet101_apart():
    assert model_loading.count_layer3_blocks(_saved_numbers_for(torchvision.models.resnet50)) == 6
    assert model_loading.count_layer3_blocks(_saved_numbers_for(torchvision.models.resnet101)) == 23


def test_build_resnet_rebuilds_a_resnet50_that_accepts_its_saved_numbers():
    saved = _saved_numbers_for(torchvision.models.resnet50)
    rebuilt = model_loading.build_resnet(saved, 3)
    rebuilt.load_state_dict(saved)
    assert len(rebuilt.layer3) == 6
    assert rebuilt.fc.out_features == 3


def test_build_resnet_rebuilds_a_resnet101_that_accepts_its_saved_numbers():
    saved = _saved_numbers_for(torchvision.models.resnet101)
    rebuilt = model_loading.build_resnet(saved, 3)
    rebuilt.load_state_dict(saved)
    assert len(rebuilt.layer3) == 23
    assert rebuilt.fc.out_features == 3


def test_build_resnet_refuses_a_model_shape_it_does_not_know():
    saved = _saved_numbers_for(torchvision.models.resnet18)
    with pytest.raises(ValueError):
        model_loading.build_resnet(saved, 3)
