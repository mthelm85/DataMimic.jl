# ─── Differential Privacy Utilities ─────────────────────────────────────────
#
# zCDP composition [Bun & Steinke 2016], Gaussian mechanism,
# exponential mechanism [McKenna et al. 2021], and PSD projection
# for Analyze-Gauss [Dwork et al. 2014].

"""
Convert (ε, δ)-DP budget to ρ-zCDP.

Uses the relation: ρ-zCDP ⟹ (ε, δ)-DP where
    ε = ρ + 2√(ρ ln(1/δ))

Inverted via the quadratic formula on √ρ.
"""
function _eps_delta_to_rho(epsilon::Float64, delta::Float64)
    c = sqrt(log(1.0 / delta))
    x = -c + sqrt(c^2 + epsilon)
    return x^2
end

"""
Gaussian mechanism noise σ for a given ρ-zCDP budget and L2 sensitivity Δ.

    ρ = Δ² / (2σ²)  ⟹  σ = Δ / √(2ρ)
"""
function _rho_to_sigma(rho::Float64, sensitivity::Float64)
    return sensitivity / sqrt(2.0 * rho)
end

"""
Sample one index from `1:length(scores)` via the exponential mechanism.

Selects index `i` with probability ∝ exp(ε × score[i] / (2Δ)).
"""
function _exponential_mechanism(scores::Vector{Float64}, epsilon::Float64,
                                sensitivity::Float64, rng::AbstractRNG)
    log_w = (epsilon / (2.0 * sensitivity)) .* scores
    log_w .-= maximum(log_w)          # numerical stability
    w = exp.(log_w)
    w ./= sum(w)
    return StatsBase.sample(rng, 1:length(scores), StatsBase.Weights(w))
end

"""
Project a symmetric matrix to the nearest positive semi-definite matrix
by clamping eigenvalues below `min_eig`.
"""
function _project_psd(M::Matrix{Float64}; min_eig::Float64 = 1e-6)
    S = LinearAlgebra.Symmetric((M + M') / 2)
    F = LinearAlgebra.eigen(S)
    vals = max.(F.values, min_eig)
    return F.vectors * LinearAlgebra.Diagonal(vals) * F.vectors'
end

"""
Project a matrix to a valid correlation matrix (PSD + unit diagonal).
"""
function _project_correlation(M::Matrix{Float64}; min_eig::Float64 = 1e-6)
    P = _project_psd(M; min_eig = min_eig)
    d = LinearAlgebra.diag(P)
    D_inv = LinearAlgebra.Diagonal(1.0 ./ sqrt.(d))
    C = D_inv * P * D_inv
    C = (C + C') / 2                  # re-symmetrize
    C = _project_psd(C; min_eig = min_eig)
    # Force exact unit diagonal and bit-exact symmetry so that
    # cholesky / Copulas.GaussianCopula accept the matrix.
    n = size(C, 1)
    for i in 1:n
        C[i, i] = 1.0
        for j in (i + 1):n
            C[j, i] = C[i, j]         # copy upper → lower
        end
    end
    return C
end

"""
Find which bin a value falls into given sorted `edges`.

Uses `searchsortedlast`: the largest index `i` where `edges[i] ≤ val`,
clamped to `[1, k]`.
"""
function _find_bin(val::Float64, edges::Vector{Float64})
    k = length(edges) - 1
    idx = searchsortedlast(edges, val)
    return clamp(idx, 1, k)
end
