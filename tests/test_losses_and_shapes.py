import pytest

torch = pytest.importorskip("torch")

from auralguard.models.aasist_backend import AASISTBackend
from auralguard.models.fusion import GatedCrossAttentionFusion
from auralguard.models.losses import OCSoftmax, SupConLoss


def test_ocsoftmax_shapes_and_score_direction():
    head = OCSoftmax(feat_dim=32)
    z = torch.randn(8, 32)
    labels = torch.tensor([0, 1] * 4)
    loss, score = head(z, labels)
    assert loss.ndim == 0
    assert score.shape == (8,)
    # inference path (no labels)
    _, score2 = head(z)
    assert score2.shape == (8,)


def test_supcon_nonnegative():
    loss_fn = SupConLoss(temperature=0.1)
    feats = torch.randn(16, 64)
    labels = torch.randint(0, 2, (16,))
    loss = loss_fn(feats, labels)
    assert loss.item() >= 0.0


def test_fusion_output_shape():
    fuse = GatedCrossAttentionFusion(dim=16, heads=2)
    fa = torch.randn(4, 16, 50)
    fb = torch.randn(4, 16, 37)
    out = fuse(fa, fb)
    assert out.shape[0] == 4 and out.shape[1] == 16


def test_backend_embedding_shape():
    back = AASISTBackend(in_dim=16, gat_dims=(12, 8), embed_dim=24)
    x = torch.randn(4, 16, 100)
    emb = back(x)
    assert emb.shape == (4, 24)
