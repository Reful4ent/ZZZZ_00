import hashlib

import numpy as np
import pytest

from metric.embed import EmbeddingSet, embed_split, load_embeddings, save_embeddings
from metric.errors import RunStop
from tests.metric.smoke.fixtures import make_sop_like, smoke_params

pytestmark = pytest.mark.unit


class CountingMethod:
    method_id = "fake"

    def __init__(self):
        self.w = np.ones(3, np.float32)
        self.calls = 0

    def embed_images(self, x):
        self.calls += 1
        return (x.mean(axis=(1, 2)) * self.w).astype(np.float32)

    def weights_hash(self):
        return hashlib.sha256(self.w.tobytes()).hexdigest()


def _setup():
    images, store = make_sop_like(n_classes=4, n_val_classes=3)
    return images, store, smoke_params()


def test_npz_roundtrip(tmp_path):
    es = EmbeddingSet(np.random.rand(3, 4).astype(np.float32), np.array(["b", "a", "c"]),
                      "w" * 64, "validation", "none", "fake", "m1", "pk")
    save_embeddings(tmp_path / "e.npz", es)
    r = load_embeddings(tmp_path / "e.npz")
    np.testing.assert_array_equal(r.embeddings, es.embeddings)
    assert list(r.image_ids) == ["b", "a", "c"]
    assert (r.weights_hash, r.split, r.degradation_id, r.method_id, r.manifest_hash, r.preprocess_key) == \
        ("w" * 64, "validation", "none", "fake", "m1", "pk")


def test_cache_hit_and_invalidate(tmp_path):
    images, store, p = _setup()
    m = CountingMethod()
    a = embed_split(m, images, "validation", store, p, "none", 0, tmp_path, "mh")
    n = m.calls
    assert n > 0
    ids = images[images.split == "validation"].image_id.sort_values().tolist()
    assert list(a.image_ids) == ids and a.embeddings.shape == (len(ids), 3)
    b = embed_split(m, images, "validation", store, p, "none", 0, tmp_path, "mh")
    assert m.calls == n
    np.testing.assert_array_equal(a.embeddings, b.embeddings)
    m.w = m.w * 2
    c = embed_split(m, images, "validation", store, p, "none", 0, tmp_path, "mh")
    assert m.calls > n and len(list(tmp_path.glob("*.npz"))) == 2
    np.testing.assert_allclose(c.embeddings, a.embeddings * 2, rtol=1e-6)
    n = m.calls  # manifest change invalidates too
    embed_split(m, images, "validation", store, p, "none", 0, tmp_path, "other")
    assert m.calls > n


def test_degradation_only_queries(tmp_path):
    images, store, p = _setup()
    m = CountingMethod()
    a = embed_split(m, images, "validation", store, p, "none", 0, tmp_path, "mh")
    embed_split(m, images, "validation", store, p, "bright_+0.1", 0, tmp_path, "mh")
    m2 = CountingMethod()
    b = embed_split(m2, images, "validation", store, p, "none", 0, tmp_path / "x", "mh")
    np.testing.assert_array_equal(a.embeddings, b.embeddings)


def test_nonfinite_raises(tmp_path):
    images, store, p = _setup()
    m = CountingMethod()
    m.w = np.full(3, np.nan, np.float32)
    with pytest.raises(RunStop) as e:
        embed_split(m, images, "validation", store, p, "none", 0, tmp_path, "mh")
    assert e.value.failure_type == "INVALID_EMBEDDING_NORM"
    assert not list(tmp_path.glob("*.npz"))


def test_preprocess_key_in_cache(tmp_path):
    images, store, p = _setup()
    m = CountingMethod()
    run = lambda deg, seed, pp=p: embed_split(m, images, "validation", store, pp, deg, seed, tmp_path, "mh")
    run("none", 0)
    n = m.calls
    p2 = {**p, "input": {**p["input"], "crop": p["input"]["crop"] - 8}}
    run("none", 0, p2)
    assert m.calls > n
    for deg in ("none", "bright_+0.1"):
        run(deg, 0)
        n = m.calls
        run(deg, 1)
        assert m.calls == n  # seed irrelevant
    run("noise_0.03", 0)
    n = m.calls
    run("noise_0.03", 0)
    assert m.calls == n
    run("noise_0.03", 1)
    assert m.calls > n
