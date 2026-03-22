"""Tests for src/model.py."""

import torch

from src.model import InferenceModel, ModelConfig, load_model


class TestModelConfig:
    def test_frozen(self):
        config = ModelConfig()
        try:
            config.batch_size = 64  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass

    def test_defaults(self):
        config = ModelConfig()
        assert config.input_channels == 3
        assert config.input_size == 224
        assert config.num_classes == 1000
        assert config.batch_size == 128


class TestInferenceModel:
    def test_features_output_shape(self):
        model = InferenceModel(num_classes=1000)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.inference_mode():
            feat = model.features(x)
            feat = torch.flatten(feat, 1)
        assert feat.shape == (4, 2048)

    def test_classifier_output_shape(self):
        model = InferenceModel(num_classes=1000)
        model.eval()
        feat = torch.randn(4, 2048)
        with torch.inference_mode():
            logits = model.classifier(feat)
        assert logits.shape == (4, 1000)

    def test_full_forward_shape(self):
        model = InferenceModel(num_classes=1000)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (2, 1000)


class TestLoadModel:
    def test_returns_eval_mode_on_cpu(self):
        config = ModelConfig(device="cpu")
        model, device = load_model(config)
        assert device == torch.device("cpu")
        assert not model.training
