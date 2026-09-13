"""
Tests for finetune_model.py, checking layer freezing and that augmentation only applies to
the training transform, not the evaluation one.
"""

import sys
from pathlib import Path

from torchvision import transforms

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


def test_train_transform_has_augmentation():
    transform_types = [type(t) for t in finetune_model.TRAIN_TRANSFORM.transforms]
    assert transforms.RandomHorizontalFlip in transform_types
    assert transforms.RandomResizedCrop in transform_types
    assert transforms.ColorJitter in transform_types


def test_eval_transform_has_no_augmentation():
    transform_types = [type(t) for t in finetune_model.EVAL_TRANSFORM.transforms]
    assert transforms.RandomHorizontalFlip not in transform_types
    assert transforms.RandomResizedCrop not in transform_types
    assert transforms.ColorJitter not in transform_types
