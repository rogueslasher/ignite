from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ignite.metrics.clustering._base import _ClusteringMetricBase

__all__ = ["DunnIndex"]


def _dunn_index(features: Tensor, labels: Tensor) -> float:
    from sklearn.metrics import pairwise_distances

    np_features = features.cpu().numpy()
    np_labels = labels.cpu().numpy()

    unique_labels = torch.unique(labels).cpu().numpy()
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        raise ValueError("Dunn Index requires at least 2 clusters.")

    # Compute intra-cluster diameters
    intra_diameters = []
    for label in unique_labels:
        cluster_points = np_features[np_labels == label]
        if len(cluster_points) < 2:
            intra_diameters.append(0.0)
        else:
            dists = pairwise_distances(cluster_points)
            intra_diameters.append(dists.max())

    # Compute inter-cluster distances
    inter_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = np_features[np_labels == unique_labels[i]]
            cluster_j = np_features[np_labels == unique_labels[j]]
            dists = pairwise_distances(cluster_i, cluster_j)
            inter_distances.append(dists.min())

    max_intra = max(intra_diameters)

    if max_intra == 0:
        raise ValueError("Maximum intra-cluster diameter is zero. Check your cluster assignments.")

    return min(inter_distances) / max_intra


class DunnIndex(_ClusteringMetricBase):
    r"""Calculates the
    `Dunn Index <https://en.wikipedia.org/wiki/Dunn_index>`_.

    The Dunn Index evaluates clustering quality by measuring the ratio
    of the minimum inter-cluster distance to the maximum intra-cluster diameter.

    A higher Dunn Index indicates better defined clusters (compact and well-separated).

    More details can be found
    `here <https://en.wikipedia.org/wiki/Dunn_index>`_.

    The computation of this metric is implemented with
    `sklearn.metrics.pairwise_distances
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`_.

    - ``update`` must receive output of the form ``(features, labels)``
      or ``{'features': features, 'labels': labels}``.
    - `features` and `labels` must be of same shape `(B, D)` and `(B,)`.

    Parameters are inherited from ``EpochMetric.__init__``.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(features, labels)``
            or ``{'features': features, 'labels': labels}``.
        check_compute_fn: if True, ``compute_fn`` is run on the first batch of data to ensure there are no
            issues. If issues exist, user is warned that there might be an issue with the ``compute_fn``.
            Default, True.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in format of
        ``(features, labels)`` or ``{'features': features, 'labels': labels, ...}``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = DunnIndex()
            metric.attach(default_evaluator, "dunn_index")
            X = torch.tensor([
                    [-1.04, -0.71, -1.42, -0.28, -0.43],
                    [0.47, 0.96, -0.43, 1.57, -2.24],
                    [-0.62, -0.29, 0.10, -0.72, -1.69],
                    [0.96, -0.77, 0.60, -0.89, 0.49],
                    [-1.33, -1.53, 0.25, -1.60, -2.0],
                    [-0.63, -0.55, -1.03, -0.89, -0.77],
                    [-0.26, -1.67, -0.24, -1.33, -0.40],
                    [-0.20, -1.34, -0.52, -1.55, -1.50],
                    [2.68, 1.13, 2.51, 0.80, 0.92],
                    [0.33, 2.88, 1.35, -0.56, 1.71]
            ])
            Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
            state = default_evaluator.run([{"features": X, "labels": Y}])
            print(state.metrics["dunn_index"])

        .. testoutput::

            0.2164...

    .. versionadded:: 0.6.1
    """

    def __init__(
        self,
        output_transform: Callable[..., Any] = lambda x: x,
        check_compute_fn: bool = True,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        try:
            from sklearn.metrics import pairwise_distances  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scikit-learn to be installed.")

        super().__init__(_dunn_index, output_transform, check_compute_fn, device, skip_unrolling)
