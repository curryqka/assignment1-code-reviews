"""
CSE 242 Machine Learning - Homework 1, Part 3 (PCA)

Implements Principal Component Analysis (PCA) from scratch with NumPy and
produces every visualization required by Questions Q2-Q4:

    Q2 - Generate a synthetic 2D dataset of two correlated Gaussian features.
    Q3 - Center the data, estimate the covariance matrix, compute its
         eigendecomposition, and select the principal components.
    Q4 - Compute per-component and cumulative explained-variance ratios.

Assumptions:
    - The input data matrix has shape (n_samples, n_features), i.e. rows are
      independent observations and columns are features.
    - The covariance matrix of the generating Gaussian must be symmetric
      positive-semidefinite (a valid covariance).
    - The unbiased covariance estimator 1/(n_samples - 1) requires at least
      two samples.

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend: save figures instead of displaying
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration constants (kept here instead of being scattered as hardcoded
# values inside the functions below).
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_SAMPLES: int = 300

# Generating distribution: feature 1 has variance 4.0, feature 2 has
# variance 1.5, and their covariance is 1.5, giving a Pearson correlation of
# rho = 1.5 / sqrt(4.0 * 1.5) ~= 0.61 (moderately strong positive correlation).
FEATURE_MEANS: np.ndarray = np.array([0.0, 0.0])
FEATURE_COVARIANCE: np.ndarray = np.array([[4.0, 1.5],
                                           [1.5, 1.5]])

FIGURE_RESOLUTION_DPI: int = 150
OUTPUT_DIRECTORY: Path = Path(".")
ARROW_SCALE: float = 3.0
VARIANCE_THRESHOLD: float = 0.9  # 90% reference line for the cumulative plot

# Colors for the two principal components / bars.
COLOR_PC1: str = "#d62728"  # red
COLOR_PC2: str = "#1f77b4"  # blue
COLOR_PC3: str = "#2ca02c"  # green (cumulative curve)


def generate_correlated_gaussian_data(
    feature_means: Sequence[float],
    feature_covariance: np.ndarray,
    n_samples: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Draw samples from a multivariate Gaussian distribution.

    Args:
        feature_means: 1-D array of shape (n_features,) holding the mean of
            each feature.
        feature_covariance: 2-D symmetric positive-semidefinite array of
            shape (n_features, n_features).
        n_samples: Positive integer number of observations to draw.
        random_generator: A ``numpy.random.Generator`` for reproducibility.

    Returns:
        data: Array of shape (n_samples, n_features) with one observation per
        row.

    Raises:
        TypeError: If ``n_samples`` is not an integer.
        ValueError: If the sample count is non-positive or the mean/covariance
            arrays have inconsistent or invalid shapes.
    """
    if not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples must be an integer.")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    means = np.asarray(feature_means, dtype=float)
    covariance = np.asarray(feature_covariance, dtype=float)

    if means.ndim != 1:
        raise ValueError("feature_means must be a 1-D array.")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("feature_covariance must be a square 2-D array.")
    if covariance.shape[0] != means.shape[0]:
        raise ValueError("feature_covariance dimensions must match the number "
                         "of features in feature_means.")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("feature_covariance must be symmetric.")

    return random_generator.multivariate_normal(
        means, covariance, size=int(n_samples)
    )


def center_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract the per-feature mean from every observation.

    Centering is required so that the covariance matrix measures spread around
    the origin: Cov(X) = E[(X - E[X])(X - E[X])^T].

    Args:
        data: Array of shape (n_samples, n_features).

    Returns:
        A tuple ``(centered_data, feature_means)`` where ``centered_data`` has
        the same shape as ``data`` and each column sums to zero.

    Raises:
        ValueError: If ``data`` is not a non-empty 2-D array.
    """
    data_array = np.asarray(data, dtype=float)
    if data_array.ndim != 2:
        raise ValueError("data must be a 2-D array of shape "
                         "(n_samples, n_features).")
    if data_array.shape[0] == 0:
        raise ValueError("data must contain at least one sample (got 0 rows).")

    feature_means = data_array.mean(axis=0)
    centered_data = data_array - feature_means
    return centered_data, feature_means


def compute_covariance_matrix(centered_data: np.ndarray) -> np.ndarray:
    """Compute the unbiased sample covariance matrix.

    Uses Sigma = (1 / (N - 1)) X_c^T X_c, where X_c is the centered data
    matrix of shape (n_samples, n_features). The ``N - 1`` denominator
    (Bessel's correction) requires at least two samples.

    Args:
        centered_data: Centered array of shape (n_samples, n_features) with
            n_samples >= 2.

    Returns:
        Symmetric array of shape (n_features, n_features).

    Raises:
        ValueError: If fewer than two samples are provided.
    """
    if centered_data.ndim != 2:
        raise ValueError("centered_data must be a 2-D array.")
    n_samples = centered_data.shape[0]
    if n_samples < 2:
        raise ValueError("At least 2 samples are required for the unbiased "
                         "covariance estimator (got %d)." % n_samples)

    return (centered_data.T @ centered_data) / (n_samples - 1)


def principal_component_analysis(
    data: np.ndarray,
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run PCA from scratch.

    Pipeline: center -> covariance -> eigendecomposition -> order by decreasing
    eigenvalue -> keep the top ``n_components`` eigenvectors.

    Args:
        data: Array of shape (n_samples, n_features) with n_samples >= 2.
        n_components: Number of leading principal components to return.
            If ``None``, all components are returned.

    Returns:
        Tuple ``(components, eigenvalues, feature_means)``:
            components: Array of shape (n_features, k) whose columns are the
                orthonormal principal component directions.
            eigenvalues: Array of shape (k,) holding the variance along each
                returned component, in decreasing order.
            feature_means: Array of shape (n_features,), the subtracted mean.

    Raises:
        ValueError: If the data is malformed or ``n_components`` is out
            of range.
    """
    centered_data, feature_means = center_data(data)
    covariance_matrix = compute_covariance_matrix(centered_data)

    # eigh exploits symmetry (more stable than eig); eigenvalues ascending.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_order]
    eigenvectors = eigenvectors[:, descending_order]

    # Round tiny negative eigenvalues caused by floating-point noise to zero.
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

    n_features = data.shape[1]
    if n_components is None:
        n_components = n_features
    if not isinstance(n_components, (int, np.integer)):
        raise TypeError("n_components must be a positive integer.")
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer "
                         "(0 iterations/components are not meaningful).")
    if n_components > n_features:
        raise ValueError("n_components (%d) cannot exceed the number of "
                         "features (%d)." % (n_components, n_features))

    components = eigenvectors[:, :n_components]
    selected_eigenvalues = eigenvalues[:n_components]
    return components, selected_eigenvalues, feature_means


def project_data(
    data: np.ndarray,
    components: np.ndarray,
    feature_means: np.ndarray,
) -> np.ndarray:
    """Project data onto the given principal components.

    Args:
        data: Array of shape (n_samples, n_features).
        components: Array of shape (n_features, k) whose columns are unit
            component vectors.
        feature_means: Array of shape (n_features,), the centering mean.

    Returns:
        scores: Array of shape (n_samples, k), the coordinate of each sample
        along every component.
    """
    return (np.asarray(data, dtype=float) - feature_means) @ components


def explained_variance_ratio(
    eigenvalues: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the per-component and cumulative explained-variance ratios.

    Args:
        eigenvalues: 1-D array of non-negative component variances.

    Returns:
        Tuple ``(ratios, cumulative_ratios)``. Both sum to one across the full
        set of components.

    Raises:
        ZeroDivisionError: If the total variance is zero (degenerate,
            non-spread data), since ratios would be undefined.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    total_variance = eigenvalues.sum()
    if total_variance <= 0.0:
        raise ZeroDivisionError(
            "Total variance is zero; explained-variance ratios are undefined "
            "(all features appear constant)."
        )
    ratios = eigenvalues / total_variance
    return ratios, np.cumsum(ratios)


def save_current_figure(filename: str) -> None:
    """Finalize layout and save the current Matplotlib figure to disk."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / filename, dpi=FIGURE_RESOLUTION_DPI)
    plt.close()


def plot_data_scatter(data: np.ndarray, filename: str) -> None:
    """Save a scatter plot of the raw 2D dataset (Q2 visualization)."""
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(data[:, 0], data[:, 1], alpha=0.6, s=18, edgecolors="none")
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.set_title("Synthetic 2D Correlated Gaussian Dataset")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    save_current_figure(filename)


def plot_principal_components(
    data: np.ndarray,
    components: np.ndarray,
    eigenvalues: np.ndarray,
    feature_means: np.ndarray,
    filename: str,
) -> None:
    """Overlay the PC directions on the data (Q3 visualization)."""
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(data[:, 0], data[:, 1], alpha=0.6, s=18,
                 edgecolors="none", label="Data")
    axis.arrow(
        feature_means[0], feature_means[1],
        components[0, 0] * ARROW_SCALE, components[1, 0] * ARROW_SCALE,
        head_width=0.25, head_length=0.25, fc=COLOR_PC1, ec=COLOR_PC1, lw=2,
        label="PC1 ($\\lambda_1$=%.2f)" % eigenvalues[0],
    )
    axis.arrow(
        feature_means[0], feature_means[1],
        components[0, 1] * ARROW_SCALE, components[1, 1] * ARROW_SCALE,
        head_width=0.25, head_length=0.25, fc=COLOR_PC2, ec=COLOR_PC2, lw=2,
        label="PC2 ($\\lambda_2$=%.2f)" % eigenvalues[1],
    )
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.set_title("Data with Principal Component Directions")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    save_current_figure(filename)


def plot_pc1_projection(
    data: np.ndarray,
    components: np.ndarray,
    feature_means: np.ndarray,
    filename: str,
) -> None:
    """Project onto PC1 only and show the 1-D reconstruction (Q3)."""
    first_component = components[:, [0]]  # keep 2-D shape (n_features, 1)
    pc1_scores = project_data(data, first_component, feature_means)
    # Reconstruct in original coordinates: x_hat = z u^T + mu.
    reconstructed = pc1_scores @ first_component.T + feature_means

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(data[:, 0], data[:, 1], alpha=0.3, s=15,
                 edgecolors="none", label="Original")
    axis.scatter(reconstructed[:, 0], reconstructed[:, 1], alpha=0.7, s=15,
                 edgecolors="none", color=COLOR_PC1,
                 label="Reconstructed (PC1 only)")
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.set_title("Projection onto the First Principal Component")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    save_current_figure(filename)


def plot_explained_variance(ratios: np.ndarray, filename: str) -> None:
    """Bar chart of variance explained by each component (Q4)."""
    fig, axis = plt.subplots(figsize=(6, 4))
    n_components = len(ratios)
    x_positions = np.arange(n_components)
    bar_colors = [COLOR_PC1, COLOR_PC2] + [COLOR_PC3] * max(
        0, n_components - 2
    )
    axis.bar(x_positions, ratios, color=bar_colors[:n_components], alpha=0.8)
    for index, ratio in enumerate(ratios):
        axis.text(index, ratio + 0.01, "%.2f%%" % (ratio * 100),
                  ha="center", fontsize=10)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(["PC%d" % (i + 1) for i in range(n_components)])
    axis.set_ylabel("Proportion of Variance Explained")
    axis.set_title("Explained Variance per Principal Component")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.3)
    save_current_figure(filename)


def plot_cumulative_variance(
    cumulative_ratios: np.ndarray,
    filename: str,
) -> None:
    """Line chart of cumulative explained variance (Q4)."""
    fig, axis = plt.subplots(figsize=(6, 4))
    n_components = len(cumulative_ratios)
    x_labels = ["PC1" if i == 0 else "PC1+PC%d" % (i + 1)
                for i in range(n_components)]
    axis.plot(x_labels, cumulative_ratios, marker="o", lw=2, color=COLOR_PC3)
    axis.axhline(VARIANCE_THRESHOLD, ls="--", color="gray", alpha=0.5,
                 label="%.0f%% threshold" % (VARIANCE_THRESHOLD * 100))
    for index, value in enumerate(cumulative_ratios):
        axis.text(index, value + 0.03, "%.2f%%" % (value * 100),
                  ha="center", fontsize=10)
    axis.set_ylabel("Cumulative Proportion of Variance")
    axis.set_title("Cumulative Explained Variance")
    axis.set_ylim(0, 1.1)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right", fontsize=9)
    save_current_figure(filename)


def main() -> None:
    """Generate data, run PCA for Q2-Q4, print results, and save figures."""
    random_generator = np.random.default_rng(seed=RANDOM_SEED)

    # --- Q2: synthetic correlated Gaussian dataset -------------------------
    data = generate_correlated_gaussian_data(
        FEATURE_MEANS, FEATURE_COVARIANCE, N_SAMPLES, random_generator
    )
    plot_data_scatter(data, "pca_scatter.png")

    # --- Q3: PCA from scratch ----------------------------------------------
    components, eigenvalues, feature_means = principal_component_analysis(data)
    covariance_matrix = compute_covariance_matrix(center_data(data)[0])

    print("Covariance matrix:\n", covariance_matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvector 1 (PC1):", components[:, 0])
    print("Eigenvector 2 (PC2):", components[:, 1])

    plot_principal_components(
        data, components, eigenvalues, feature_means, "pca_directions.png"
    )
    plot_pc1_projection(
        data, components, feature_means, "pca_projection.png"
    )

    # --- Q4: explained variance --------------------------------------------
    ratios, cumulative_ratios = explained_variance_ratio(eigenvalues)
    print("Explained variance ratio:", ratios)
    print("Cumulative explained variance:", cumulative_ratios)

    plot_explained_variance(ratios, "pca_variance.png")
    plot_cumulative_variance(cumulative_ratios, "pca_cumulative.png")

    print("All figures generated.")


if __name__ == "__main__":
    main()
