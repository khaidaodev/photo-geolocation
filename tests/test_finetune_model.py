"""
Tests for finetune_model.py: layer freezing, augmentation setup, learning rate, best-epoch
tracking, and early stopping logic.
"""

import sys
from pathlib import Path

from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import finetune_model


def test_layer3_layer4_and_fc_are_trainable():
    model = finetune_model.build_model(num_classes=20)

    trainable_prefixes = ("layer3", "layer4", "fc")
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
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


def test_best_epoch_picks_highest_accuracy():
    epoch_num, acc = finetune_model.best_epoch([0.10, 0.14, 0.12, 0.09])
    assert epoch_num == 2
    assert acc == 0.14


def test_best_epoch_handles_last_epoch_being_best():
    epoch_num, acc = finetune_model.best_epoch([0.10, 0.11, 0.15])
    assert epoch_num == 3
    assert acc == 0.15


def test_learning_rate_is_lower_than_previous_attempt():
    assert finetune_model.LEARNING_RATE == 1e-5


def test_should_stop_early_false_within_patience():
    accuracies = [0.10, 0.14, 0.12, 0.11]
    assert not finetune_model.should_stop_early(accuracies, patience=5)


def test_should_stop_early_true_after_patience_exceeded():
    accuracies = [0.10, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06]
    assert finetune_model.should_stop_early(accuracies, patience=5)


def test_should_stop_early_false_with_too_few_epochs():
    assert not finetune_model.should_stop_early([0.10, 0.14], patience=5)
