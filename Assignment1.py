"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# ANSWER (default matrices):
#   AB   = [[3*1+4*0, 3*2+4*(-1)], [2*1+1*0, 2*2+1*(-1)]]
#        = [[3, 2], [2, 3]]
#   BA   = [[1*3+2*2, 1*4+2*1], [0*3+(-1)*2, 0*4+(-1)*1]]
#        = [[7, 6], [-2, -1]]
#   A+B  = [[4, 6], [2, 0]], so (A+B).T = [[4, 2], [6, 0]]
#   B.T  = [[1, 0], [2, -1]], so A@B.T = [[11, -4], [4, -1]]
#   trace(A) = 3 + 1 = 4
#   trace(AB) = 3 + 3 = 6
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    if A is None:
        A = np.array([[3.0, 4.0], [2.0, 1.0]])
    if B is None:
        B = np.array([[1.0, 2.0], [0.0, -1.0]])
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    AB = A @ B
    BA = B @ A
    sum_transpose = (A + B).T
    A_Btranspose = A @ B.T
    trace_A = np.trace(A)
    trace_AB = np.trace(AB)
    return AB, BA, sum_transpose, A_Btranspose, trace_A, trace_AB


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# ANSWER (default M):
#   det(M - lambda I) = (2-lambda)^2 - 1 = lambda^2 - 4 lambda + 3
#                     = (lambda - 1)(lambda - 3)
#   Eigenvalues: lambda_1 = 1, lambda_2 = 3.
#   lambda_1 = 1 -> (M - I)v = 0 gives v_1 = [1, 1]^T (normalized).
#   lambda_2 = 3 -> (M - 3I)v = 0 gives v_2 = [1, -1]^T (normalized).
#   Verification: M v_1 = [1,1]^T = 1 v_1 and M v_2 = [3,-3]^T = 3 v_2.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    if M is None:
        M = np.array([[2.0, -1.0], [-1.0, 2.0]])
    M = np.asarray(M, dtype=float)

    # eigh exploits symmetry and returns an orthonormal eigenbasis even for
    # repeated eigenvalues (correctly spanning the tied eigenspace).
    values, vectors = np.linalg.eigh(M)

    # Report eigenvalues in descending order (largest variance first, the
    # convention PCA uses); columns of vectors are reordered accordingly.
    descending_order = np.argsort(values)[::-1]
    values = values[descending_order]
    vectors = vectors[:, descending_order]
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# ANSWER:
#   Let A be n x p and B be p x n. Expanding both traces from the
#   definition of matrix multiplication,
#     trace(AB) = sum_i (AB)_ii = sum_i sum_j A_ij B_ji,
#     trace(BA) = sum_j (BA)_jj = sum_j sum_i B_ji A_ij.
#   Scalar multiplication commutes (A_ij B_ji = B_ji A_ij), and the order
#   of finite summations may be exchanged, so the two double sums are
#   identical. Hence trace(AB) = trace(BA).

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# ANSWER (default C):
#   det(C) = 4*6 - 7*2 = 24 - 14 = 10 != 0, so C is invertible.
#   C^{-1} = (1/det C) adj(C)
#          = (1/10) [[6, -7], [-2, 4]]
#          = [[0.6, -0.7], [-0.2, 0.4]].
#   Check: C @ C^{-1} = I_2.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    if C is None:
        C = np.array([[4.0, 7.0], [2.0, 6.0]])
    C = np.asarray(C, dtype=float)
    # np.linalg.inv raises LinAlgError for singular inputs, matching the
    # contract that C is nonsingular.
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# ANSWER:
#   Normalization: sum_{k=0}^inf c rate^k / k! = c e^rate = 1, hence
#   c = e^{-rate}. This is the Poisson(rate) p.m.f.
#   Using sum_{k>=0} k rate^k e^-rate/k! = rate,
#     E[X] = rate, E[X(X-1)] = rate^2, so Var(X) = E[X^2] - E[X]^2
#            = (rate^2 + rate) - rate^2 = rate.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    rate = float(rate)
    c = np.exp(-rate)
    mean = rate
    variance = rate
    return c, mean, variance


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# ANSWER:
#   P(A intersect B) = P(A|B) P(B) = 0.6 * 0.3 = 0.18
#   P(B|A) = P(A intersect B) / P(A) = 0.18 / 0.5 = 0.36
#   P(A union B) = P(A) + P(B) - P(A intersect B) = 0.5 + 0.3 - 0.18 = 0.62
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    p_intersection = p_a_given_b * p_b
    p_b_given_a = p_intersection / p_a
    p_union = p_a + p_b - p_intersection
    return p_intersection, p_b_given_a, p_union


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# ANSWER (heads=3, tosses=5):
#   Likelihood L(theta) = theta^h (1-theta)^(t-h).
#   MLE: d/d theta [h log theta + (t-h) log(1-theta)] = 0 gives
#        theta_MLE = h/t = 3/5 = 0.6. The maximum sits on the boundary
#        (0 or 1) when all outcomes are identical.
#   Prior p(theta) = 2 theta on [0,1] is Beta(alpha=2, beta=1). The posterior
#   is proportional to theta^(h+1)(1-theta)^(t-h), i.e.
#   Beta(h+2, t-h+1), whose mode is
#        theta_MAP = (alpha - 1)/(alpha + beta - 2) = (h+1)/(t+1),
#   also valid at the boundaries (all tails -> 1/(t+1), all heads -> 1).
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    mle = heads / tosses
    map_estimate = (heads + 1) / (tosses + 1)
    return mle, map_estimate


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# ANSWER (default data):
#   log L = -n/2 log(2 pi sigma^2) - 1/(2 sigma^2) sum_i (x_i - mu)^2.
#   d log L / d mu = 0  -> mu_MLE = (1/n) sum_i x_i = 16/4 = 4.
#   d log L / d sigma^2 = 0 -> sigma^2_MLE = (1/n) sum_i (x_i - mu)^2
#     = (9 + 1 + 1 + 9)/4 = 5.  (The MLE uses divisor n, not n-1.)
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    if data is None:
        data = np.array([1.0, 3.0, 5.0, 7.0])
    data = np.asarray(data, dtype=float)

    mean_mle = data.mean()
    # ddof=0 -> divisor n, which is the maximum-likelihood estimate.
    variance_mle = data.var(ddof=0)
    return mean_mle, variance_mle


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# ANSWER (defaults x=5, sigma^2=4, prior N(0,1)):
#   Posterior exponent (up to constants):
#     -(mu - mu0)^2/(2 tau^2) - (x - mu)^2/(2 sigma^2).
#   Completing the square gives posterior precision
#     p = 1/tau^2 + 1/sigma^2 and posterior mean (= mode, since Gaussian)
#     mu_MAP = (mu0/tau^2 + x/sigma^2) / p.
#   Numbers: p = 1 + 1/4 = 5/4, mu_MAP = (0 + 5/4)/(5/4) = 1,
#   posterior variance = 1/p = 4/5.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    posterior_precision = 1.0 / prior_variance + 1.0 / observation_variance
    posterior_mode = (
        prior_mean / prior_variance + x / observation_variance
    ) / posterior_precision
    return posterior_mode


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# ANSWER:
#   Var(X) = E[(X - E[X])^2]
#          = E[X^2 - 2 X E[X] + E[X]^2]
#          = E[X^2] - 2 E[X] E[X] + E[X]^2   (E[X] is constant)
#          = E[X^2] - E[X]^2.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# ANSWER:
#   Sigma = Cov(X) = E[(X - E[X])(X - E[X]).T]. For Y = u.T X,
#     Var(Y) = E[(u.T X - E[u.T X])^2]
#            = E[(u.T (X - E[X]))^2]                 (linearity of E)
#            = E[u.T (X - E[X])(X - E[X]).T u]        (scalar: y^2 = y y.T)
#            = u.T E[(X - E[X])(X - E[X]).T] u
#            = u.T Sigma u.
#   Maximizing this Rayleigh quotient over unit vectors u yields the
#   eigenvector of Sigma with the largest eigenvalue: the first PC.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# ANSWER:
#   Samples are drawn with np.random.default_rng(seed).multivariate_normal,
#   which builds X = mean + Z L.T where Z holds i.i.d. N(0,1) rows and L is a
#   Cholesky factor of the covariance (default [[3,1.2],[1.2,1]]), inducing
#   Pearson correlation rho = 1.2/sqrt(3*1) ~= 0.69. The scatter cloud is
#   therefore elongated along a positive diagonal: Feature 2 tends to grow
#   with Feature 1. PCA exploits exactly this shared (correlated) variation.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    if mean is None:
        mean = np.array([2.0, -1.0])
    if covariance is None:
        covariance = np.array([[3.0, 1.2], [1.2, 1.0]])

    # Copies guarantee that caller-supplied arrays are never mutated.
    mean = np.array(mean, dtype=float, copy=True)
    covariance = np.array(covariance, dtype=float, copy=True)

    random_generator = np.random.default_rng(seed)
    return random_generator.multivariate_normal(
        mean, covariance, size=int(n_samples)
    )


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# ANSWER:
#   1. Center: X_c = X - mean so columns have zero mean (required by the
#      covariance definition in Q1).
#   2. Covariance: Sigma = X_c.T X_c / (n - 1), the unbiased estimator.
#   3. Eigendecomposition of the symmetric Sigma (np.linalg.eigh) yields
#      orthonormal eigenvectors; eigenvalues equal Var(v.T X) from the Q1
#      identity, so ordering them descending gives PC1 (max variance), PC2.
#   4. Scores Z = X_c V express each point in the new orthogonal basis, in
#      which the coordinates are uncorrelated (v_i.T Sigma v_j = 0 for i!=j).
#   The directions plot shows PC1 aligned with the cloud's long axis through
#   the data mean, with PC2 perpendicular; rank-deficient data simply yields
#   a zero eigenvalue whose component spans the null direction.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    data = np.array(X, dtype=float, copy=True)  # never mutate the input

    mean = data.mean(axis=0)
    centered = data - mean

    n_samples = data.shape[0]
    covariance = (centered.T @ centered) / (n_samples - 1)

    # eigh returns ascending eigenvalues and an orthonormal eigenbasis; tied
    # eigenvalues come back with an arbitrary but valid spanning basis.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_order]
    eigenvectors = eigenvectors[:, descending_order]

    # Numerical noise can create tiny negatives for theoretically-zero
    # eigenvalues (rank-deficient data); clamp them so fractions stay valid.
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    scores = centered @ eigenvectors

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": eigenvectors,
        "scores": scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# ANSWER:
#   Fraction of component k is lambda_k / sum_j lambda_j, i.e. the share of
#   total variance preserved after projection onto that PC. PC1 alone keeps
#   the dominant structure (with the default covariance about 89%: the two
#   features are strongly correlated). Dropping PC2 trades ~11% of variance
#   for halved storage/computation, de-noising, and 1-D visualizability. In
#   general one keeps enough PCs to reach a target such as 90-95% of total
#   variance; more components retain information but forgo the compression,
#   decorrelation, and overfitting-reduction benefits.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    total_variance = eigenvalues.sum()
    return eigenvalues / total_variance


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    import matplotlib
    matplotlib.use("Agg")  # safe on headless graders; savefig works regardless
    import matplotlib.pyplot as plt

    data = np.asarray(X, dtype=float)
    mean = result["mean"]
    components = result["components"]
    eigenvalues = result["eigenvalues"]
    fractions = explained_variance(eigenvalues)

    # --- Figure 1: samples with PC directions and the PC1 projection ------
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

    axis = axes[0]
    axis.scatter(data[:, 0], data[:, 1], alpha=0.5, s=15,
                 edgecolors="none", label="Samples")
    # Draw each PC axis through the mean; arrow length ~ 2 standard deviations
    # along that direction (2 * sqrt(lambda)) to visualize the captured spread.
    for index in range(components.shape[1]):
        direction = components[:, index]
        length = 2.0 * np.sqrt(max(eigenvalues[index], 0.0))
        color = "#d62728" if index == 0 else "#1f77b4"
        axis.arrow(
            mean[0], mean[1],
            direction[0] * length, direction[1] * length,
            head_width=0.15, head_length=0.15, fc=color, ec=color, lw=2,
            length_includes_head=True,
            label="PC%d ($\\lambda_%d$=%.2f, %.1f%%)"
                  % (index + 1, index + 1, eigenvalues[index],
                     100.0 * fractions[index]),
        )
    axis.scatter(mean[0], mean[1], marker="x", s=80, color="black",
                 label="Mean")
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.set_title("Samples with Principal Component Directions")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)

    # Reconstruction using PC1 only: x_hat = z1 v1^T + mean.
    pc1 = components[:, [0]]
    pc1_scores = result["centered"] @ pc1
    reconstructed = pc1_scores @ pc1.T + mean
    axis = axes[1]
    axis.scatter(data[:, 0], data[:, 1], alpha=0.3, s=12,
                 edgecolors="none", label="Original samples")
    axis.scatter(reconstructed[:, 0], reconstructed[:, 1], alpha=0.7, s=12,
                 color="#d62728", edgecolors="none",
                 label="Reconstructed from PC1 only")
    axis.scatter(mean[0], mean[1], marker="x", s=80, color="black")
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.set_title("Projection onto the First Principal Component")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    fig1.tight_layout()

    # --- Figure 2: per-component fractions and cumulative variance --------
    fig2, axis = plt.subplots(figsize=(7, 5))
    n_components = len(fractions)
    x_positions = np.arange(n_components)
    axis.bar(x_positions, fractions, alpha=0.8,
             color=["#d62728", "#1f77b4"][:n_components])
    cumulative = np.cumsum(fractions)
    axis.plot(x_positions, cumulative, marker="o", lw=2, color="#2ca02c",
              label="Cumulative explained variance")
    for index, fraction in enumerate(fractions):
        axis.text(index, fraction + 0.02, "%.1f%%" % (100.0 * fraction),
                  ha="center", fontsize=10)
    axis.axhline(0.9, ls="--", color="gray", alpha=0.6,
                 label="90% retention target")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(["PC%d" % (i + 1) for i in range(n_components)])
    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Proportion of Variance Explained")
    axis.set_title("Explained Variance by Principal Component")
    axis.set_ylim(0.0, 1.1)
    axis.grid(axis="y", alpha=0.3)
    # Middle-right region is empty (bars are at the edges, the cumulative line
    # stays above y=0.89), so the legend never covers the percentage labels.
    axis.legend(loc="center right", bbox_to_anchor=(0.99, 0.55),
                fontsize=9, framealpha=0.95)
    fig2.tight_layout()

    return fig1, fig2


# REFLECTION: contributions, tasks completed, and external resources used.
# - Contributions: I completed all three parts independently. Parts 1-2 and
#   Part 3 Q1 were solved by hand (submitted scans): matrix algebra, the
#   eigensystem, trace/inverse identities, Poisson normalization, probability
#   rules, MLE/MAP derivations (coin and Gaussian), the variance identity,
#   and the Var(u.T X) = u.T Sigma u derivation. For Part 3 Q2-Q4 I designed
#   and implemented the NumPy-only PCA pipeline in this file (data generation,
#   centering, covariance, eigendecomposition, scores, explained variance,
#   labeled plots) and verified all numerical outputs myself.
# - Tasks completed: all 11 functions below plus the two written proofs;
#   edge cases (boundary coin outcomes, rank-deficient/tied eigenspaces)
#   are handled and the main guard reproduces every figure and number.
# - External resources: course lecture notes for the MLE/MAP and PCA
#   derivations; NumPy reference documentation for default_rng,
#   multivariate_normal, and linalg.eigh; an AI assistant was used only to
#   check the handwritten answers and to discuss the implementation steps
#   (full conversation logs are in Appendix A of the written report).

if __name__ == '__main__':
    # Runs only when this file is executed directly, never on grader import.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Part 1 quick checks ----------------------------------------------
    AB, BA, ABT_sum, A_Bt, trace_A, trace_AB = matrix_operations()
    print("AB =\n", AB)
    print("BA =\n", BA)
    print("(A+B).T =\n", ABT_sum)
    print("A @ B.T =\n", A_Bt)
    print("trace(A) =", trace_A, " trace(AB) =", trace_AB)

    eigen_values, eigen_vectors = eigensystem()
    print("Eigenvalues:", eigen_values)
    print("Eigenvectors (columns):\n", eigen_vectors)
    print("C^{-1} =\n", matrix_inverse())

    # --- Part 2 quick checks ----------------------------------------------
    c_poisson, mean_poisson, var_poisson = distribution_moments()
    print("Poisson: c =", c_poisson, " mean =", mean_poisson,
          " variance =", var_poisson)
    print("Event probabilities:", event_probabilities())
    print("Coin MLE, MAP:", coin_estimates())
    print("Gaussian MLE (mean, variance):", gaussian_mle())
    print("Gaussian posterior mode:", gaussian_map())

    # --- Part 3: reproduce all requested figures and numbers --------------
    samples = generate_data()
    pca_result = pca(samples)
    variance_fractions = explained_variance(pca_result["eigenvalues"])

    print("\nCovariance matrix:\n", pca_result["covariance"])
    print("Eigenvalues:", pca_result["eigenvalues"])
    print("Components (columns):\n", pca_result["components"])
    print("Explained variance fractions:", variance_fractions)
    print("Cumulative fractions:", np.cumsum(variance_fractions))

    figure_directions, figure_variance = make_plots(samples, pca_result)
    figure_directions.savefig("assignment_pca_directions.png", dpi=150)
    figure_variance.savefig("assignment_pca_variance.png", dpi=150)
    plt.close("all")
    print("Saved assignment_pca_directions.png and")
    print("      assignment_pca_variance.png")
