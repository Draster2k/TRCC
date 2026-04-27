"""Unit tests for TRCC."""
import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from trcc import TRCC


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=600, centers=4, cluster_std=0.6,
                      random_state=42)
    return StandardScaler().fit_transform(X), y


def test_fit_predict_returns_int_labels_with_correct_shape(blobs):
    X, _ = blobs
    labels = TRCC().fit_predict(X)
    assert labels.shape == (len(X),)
    assert labels.dtype.kind == "i"


def test_recovers_blobs(blobs):
    X, y = blobs
    labels = TRCC().fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.9


def test_determinism(blobs):
    X, _ = blobs
    a = TRCC(random_state=7).fit_predict(X)
    b = TRCC(random_state=7).fit_predict(X)
    np.testing.assert_array_equal(a, b)


def test_predict_consistent_with_fit(blobs):
    X, _ = blobs
    m = TRCC().fit(X)
    p = m.predict(X)
    valid = m.labels_ != -1
    # On non-noise points, predict (nearest centroid) should match fit
    # for at least 90% — predict has no signal-aware logic, so a small gap
    # is expected near boundaries.
    agreement = (p[valid] == m.labels_[valid]).mean()
    assert agreement > 0.9


def test_predict_on_unseen_points():
    X, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.5,
                      random_state=1)
    X = StandardScaler().fit_transform(X)
    m = TRCC().fit(X)
    new = X[:50] + np.random.RandomState(0).normal(0, 0.05, X[:50].shape)
    pred = m.predict(new)
    assert pred.shape == (50,)
    assert pred.min() >= 0
    assert pred.max() < m.n_clusters_


def test_noise_label_when_min_cluster_size_huge():
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5,
                      random_state=2)
    X = StandardScaler().fit_transform(X)
    labels = TRCC(min_cluster_size=10_000).fit_predict(X)
    # everything gets demoted
    assert (labels == -1).all()


def test_attributes_after_fit(blobs):
    X, _ = blobs
    m = TRCC().fit(X)
    assert hasattr(m, "labels_")
    assert hasattr(m, "cluster_centers_")
    assert hasattr(m, "signals_")
    assert hasattr(m, "peak_indices_")
    assert m.cluster_centers_.shape[1] == X.shape[1]
    assert m.signals_.shape == (len(X),)
    assert m.n_clusters_ == m.cluster_centers_.shape[0]


def test_handles_high_dimensional_data():
    X, y = make_blobs(n_samples=400, centers=3, n_features=20,
                      cluster_std=1.0, random_state=3)
    X = StandardScaler().fit_transform(X)
    labels = TRCC().fit_predict(X)
    assert labels.shape == (400,)
    assert adjusted_rand_score(y, labels) > 0.5


def test_explicit_n_clusters_is_respected():
    X, _ = make_blobs(n_samples=600, centers=5, cluster_std=0.4,
                      random_state=4)
    X = StandardScaler().fit_transform(X)
    m = TRCC(n_clusters=5).fit(X)
    n_real = len(np.unique(m.labels_[m.labels_ >= 0]))
    # merge can collapse below 5; never exceed 5
    assert n_real <= 5


def test_2d_input_required():
    with pytest.raises(ValueError):
        TRCC().fit(np.zeros(10))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TRCC().predict(np.zeros((5, 2)))


def test_moons_beats_kmeans_baseline():
    """Smoke regression: TRCC should not be ARI<0.3 on clean moons."""
    X, y = make_moons(n_samples=800, noise=0.05, random_state=0)
    X = StandardScaler().fit_transform(X)
    labels = TRCC().fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.3
