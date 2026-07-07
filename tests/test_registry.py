import pytest

torch = pytest.importorskip("torch")

from auralguard.models import available, build_model, register


def test_builtin_models_registered():
    names = available()
    for expected in ["auralguard", "lfcc_lcnn", "rawnet2", "aasist_raw"]:
        assert expected in names


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="unknown model"):
        build_model({"name": "does_not_exist"})


@pytest.mark.parametrize("name", ["rawnet2", "aasist_raw"])
def test_baseline_contract(name):
    """Any registered model must honor the forward contract (score/loss dict)."""
    model = build_model({"name": name})
    wav = torch.randn(2, 16000)
    labels = torch.tensor([0, 1])

    out = model(wav)  # inference path
    assert out["score"].shape == (2,)

    out = model(wav, labels)  # training path
    assert out["loss"].ndim == 0
    out["loss"].backward()  # must be differentiable


def test_lfcc_lcnn_contract():
    pytest.importorskip("torchaudio")
    model = build_model({"name": "lfcc_lcnn"})
    out = model(torch.randn(2, 16000), torch.tensor([0, 1]))
    assert out["score"].shape == (2,) and out["loss"].ndim == 0


def test_register_custom_model():
    """The EXTENDING.md recipe: a user model plugs in via the decorator."""

    @register("_test_dummy")
    class Dummy(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.w = torch.nn.Linear(1, 1)

        def forward(self, wav, labels=None):
            score = self.w(wav.mean(dim=1, keepdim=True)).squeeze(1)
            out = {"score": score}
            if labels is not None:
                out["loss"] = score.mean()
            return out

    m = build_model({"name": "_test_dummy"})
    assert m(torch.randn(3, 16000))["score"].shape == (3,)


def test_arch_override():
    """`arch` lets many configs share one architecture (e.g. b5 reuses auralguard)."""
    m = build_model({"name": "some_variant", "arch": "aasist_raw"}).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 16000))["score"].shape == (1,)
