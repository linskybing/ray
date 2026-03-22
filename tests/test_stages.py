"""Tests for src/stages.py."""

import numpy as np
import torch
import torch.nn as nn

from src.stages import (
    stage1_preprocess,
    stage2_extract_and_infer,
    stage2_feature_extract,
    stage3_inference,
    stage4_postprocess,
)


class TestStage1:
    def test_output_shape(self):
        result = stage1_preprocess(batch_size=8)
        assert result.shape == (8, 3, 224, 224)

    def test_output_dtype(self):
        result = stage1_preprocess(batch_size=4)
        assert result.dtype == np.float32

    def test_normalized_range(self):
        result = stage1_preprocess(batch_size=16)
        # After ImageNet normalization, values should not all be in [0,1]
        assert result.min() < 0.0 or result.max() > 1.0

    def test_with_rng(self):
        rng = np.random.default_rng(seed=42)
        result = stage1_preprocess(batch_size=4, rng=rng)
        assert result.shape == (4, 3, 224, 224)
        assert result.dtype == np.float32

    def test_rng_deterministic(self):
        r1 = stage1_preprocess(4, rng=np.random.default_rng(seed=0))
        r2 = stage1_preprocess(4, rng=np.random.default_rng(seed=0))
        np.testing.assert_array_equal(r1, r2)


class TestStage2MergedInference:
    """Tests for the merged stage2_extract_and_infer."""

    def test_output_types(self):
        images = np.random.randn(4, 3, 224, 224).astype(np.float32)
        features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        classifier = nn.Linear(64, 1000)
        features.eval()
        classifier.eval()
        device = torch.device("cpu")

        logits_cpu, cpu_feat = stage2_extract_and_infer(
            images, features, classifier, device,
        )
        assert isinstance(logits_cpu, torch.Tensor)
        assert isinstance(cpu_feat, np.ndarray)
        assert not logits_cpu.is_cuda

    def test_logits_shape(self):
        images = np.random.randn(4, 3, 224, 224).astype(np.float32)
        features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        classifier = nn.Linear(64, 1000)
        features.eval()
        classifier.eval()
        device = torch.device("cpu")

        logits_cpu, _ = stage2_extract_and_infer(
            images, features, classifier, device,
        )
        assert logits_cpu.shape == (4, 1000)

    def test_cpu_features_shape(self):
        images = np.random.randn(4, 3, 224, 224).astype(np.float32)
        features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        classifier = nn.Linear(64, 1000)
        features.eval()
        classifier.eval()
        device = torch.device("cpu")

        _, cpu_feat = stage2_extract_and_infer(
            images, features, classifier, device,
        )
        # mean(3 channels) + std(3 channels) = 6
        assert cpu_feat.shape == (4, 6)


class TestStage2Legacy:
    """Legacy stage2_feature_extract still works for backward compat."""

    def test_output_types(self):
        images = np.random.randn(4, 3, 224, 224).astype(np.float32)
        features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        features.eval()
        device = torch.device("cpu")

        gpu_feat, cpu_feat = stage2_feature_extract(images, features, device)
        assert isinstance(gpu_feat, torch.Tensor)
        assert isinstance(cpu_feat, np.ndarray)

    def test_cpu_features_shape(self):
        images = np.random.randn(4, 3, 224, 224).astype(np.float32)
        features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        features.eval()
        device = torch.device("cpu")

        _, cpu_feat = stage2_feature_extract(images, features, device)
        assert cpu_feat.shape == (4, 6)


class TestStage3:
    def test_logits_shape(self):
        classifier = nn.Linear(64, 1000)
        classifier.eval()
        gpu_feat = torch.randn(8, 64)

        logits = stage3_inference(gpu_feat, classifier)
        assert logits.shape == (8, 1000)


class TestStage4:
    def test_returns_dict(self):
        logits = torch.randn(8, 1000)
        cpu_feat = np.random.randn(8, 6).astype(np.float32)

        result = stage4_postprocess(logits, cpu_feat)
        assert isinstance(result, dict)
        assert result["num_samples"] == 8
        assert len(result["predictions"]) == 8
        assert len(result["confidences"]) == 8

    def test_predictions_in_range(self):
        logits = torch.randn(4, 1000)
        cpu_feat = np.random.randn(4, 6).astype(np.float32)

        result = stage4_postprocess(logits, cpu_feat)
        assert all(0 <= p < 1000 for p in result["predictions"])

    def test_confidences_are_probabilities(self):
        logits = torch.randn(4, 1000)
        cpu_feat = np.random.randn(4, 6).astype(np.float32)

        result = stage4_postprocess(logits, cpu_feat)
        assert all(0.0 <= c <= 1.0 for c in result["confidences"])
