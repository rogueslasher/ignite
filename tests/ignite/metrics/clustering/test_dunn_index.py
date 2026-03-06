import numpy as np
import pytest
import torch
from torch import Tensor

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.clustering import DunnIndex


def test_zero_sample():
    with pytest.raises(NotComputableError, match="DunnIndex must have at least one example before it can be computed"):
        metric = DunnIndex()
        metric.compute()


def test_wrong_output_shape():
    wrong_features = torch.zeros(4, dtype=torch.float)
    correct_features = torch.zeros(4, 3, dtype=torch.float)
    wrong_labels = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
    correct_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    with pytest.raises(ValueError, match=r"Features should be of shape \(batch_size, n_targets\)"):
        metric = DunnIndex()
        metric.update((wrong_features, correct_labels))

    with pytest.raises(ValueError, match=r"Labels should be of shape \(batch_size, \)"):
        metric = DunnIndex()
        metric.update((correct_features, wrong_labels))


def test_wrong_output_dtype():
    wrong_features = torch.zeros(4, 3, dtype=torch.long)
    correct_features = torch.zeros(4, 3, dtype=torch.float)
    wrong_labels = torch.tensor([0, 0, 1, 1], dtype=torch.float)
    correct_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    with pytest.raises(ValueError, match=r"Incoherent types between input features and stored features"):
        metric = DunnIndex()
        metric.update((correct_features, correct_labels))
        metric.update((wrong_features, correct_labels))

    with pytest.raises(ValueError, match=r"Incoherent types between input labels and stored labels"):
        metric = DunnIndex()
        metric.update((correct_features, correct_labels))
        metric.update((correct_features, wrong_labels))


def _compute_dunn_index(features: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import pairwise_distances

    unique_labels = np.unique(labels)

    intra_diameters = []
    for label in unique_labels:
        cluster_points = features[labels == label]
        if len(cluster_points) < 2:
            intra_diameters.append(0.0)
        else:
            dists = pairwise_distances(cluster_points)
            intra_diameters.append(dists.max())

    inter_distances = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            ci = features[labels == unique_labels[i]]
            cj = features[labels == unique_labels[j]]
            dists = pairwise_distances(ci, cj)
            inter_distances.append(dists.min())

    max_intra = max(intra_diameters)

    if max_intra == 0:
        raise ValueError("Maximum intra-cluster diameter is zero. Check your cluster assignments.")

    return min(inter_distances) / max_intra


@pytest.fixture(params=list(range(2)))
def test_case(request):
    N = 100
    NDIM = 10
    BS = 10

    # well-clustered case
    random_order = torch.from_numpy(np.random.permutation(N * 3))
    x1 = torch.cat(
        [
            torch.normal(-5.0, 1.0, size=(N, NDIM)),
            torch.normal(5.0, 1.0, size=(N, NDIM)),
            torch.normal(0.0, 1.0, size=(N, NDIM)),
        ]
    ).float()[random_order]
    y1 = torch.tensor([0] * N + [1] * N + [2] * N, dtype=torch.long)[random_order]

    # poorly-clustered case
    x2 = torch.cat(
        [
            torch.normal(-1.0, 1.0, size=(N, NDIM)),
            torch.normal(0.0, 1.0, size=(N, NDIM)),
            torch.normal(1.0, 1.0, size=(N, NDIM)),
        ]
    ).float()
    y2 = torch.from_numpy(np.random.choice(3, size=N * 3)).long()

    return [
        (x1, y1, BS),
        (x2, y2, BS),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_integration(n_times: int, test_case: tuple[Tensor, Tensor, int], available_device):
    features, labels, batch_size = test_case

    np_features = features.numpy()
    np_labels = labels.numpy()

    def update_fn(engine: Engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        feature_batch = np_features[idx : idx + batch_size]
        label_batch = np_labels[idx : idx + batch_size]
        return torch.from_numpy(feature_batch), torch.from_numpy(label_batch)

    engine = Engine(update_fn)
    m = DunnIndex(device=available_device)
    assert m._device == torch.device(available_device)
    m.attach(engine, "dunn_index")

    data = list(range(np_features.shape[0] // batch_size))
    s = engine.run(data, max_epochs=1).metrics["dunn_index"]

    np_ans = _compute_dunn_index(np_features, np_labels)
    assert pytest.approx(np_ans, rel=1e-5) == s
