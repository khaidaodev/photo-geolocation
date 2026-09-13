"""
Tests for finetune_model.py's build_model(), checking the right layers are frozen/unfrozen
without needing to actually train anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import finetune_model


def test_only_layer4_and_fc_are_trainable():
    model = finetune_model.build_model(num_classes=20)

    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_final_layer_matches_num_classes():
    model = finetune_model.build_model(num_classes=20)
    assert model.fc.out_features == 20
