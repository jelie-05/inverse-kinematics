import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel):
    """
    Empirical Maximum Mean Discrepancy (MMD) with detailed explanation.
    
    MMD measures the distance between two probability distributions P and Q
    by comparing their embeddings in a reproducing kernel Hilbert space (RKHS).
    
    The MMD^2 between distributions P and Q is defined as:
    MMD^2(P, Q) = E[k(X, X')] + E[k(Y, Y')] - 2E[k(X, Y)]
    
    Where:
    - X, X' are samples from distribution P
    - Y, Y' are samples from distribution Q  
    - k(·,·) is a kernel function
    - E[·] denotes expectation
    
    For finite samples, we estimate this using sample averages.

    Args:
        x: first sample, distribution P (shape: [n_samples, n_features])
        y: second sample, distribution Q (shape: [n_samples, n_features])
        kernel: kernel type such as "multiscale" or "rbf"
    
    Returns:
        MMD value (scalar tensor) - lower values indicate more similar distributions
    """
    
    # Step 1: Compute pairwise inner products
    # xx[i,j] = <x_i, x_j>, yy[i,j] = <y_i, y_j>, zz[i,j] = <x_i, y_j>
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    
    # Step 2: Extract diagonal elements (squared norms)
    # rx[i,j] = ||x_i||^2, ry[i,j] = ||y_i||^2 (broadcasted to matrix form)
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    # Step 3: Compute squared Euclidean distances using the identity:
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>
    dxx = rx.t() + rx - 2. * xx  # Squared distances between samples in x
    dyy = ry.t() + ry - 2. * yy  # Squared distances between samples in y  
    dxy = rx.t() + ry - 2. * zz  # Squared distances between x and y samples

    # Step 4: Initialize kernel matrices
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    # Step 5: Apply kernel function to compute k(x_i, x_j), k(y_i, y_j), k(x_i, y_j)
    if kernel == "multiscale":
        # Multiscale kernel: sum of inverse multiquadratic kernels with different bandwidths
        # k(x, y) = sum_σ (σ^2 / (σ^2 + ||x-y||^2))
        # This kernel is characteristic (can distinguish any two different distributions)
        bandwidth_range = [0.1, 0.2, 0.5, 1.0, 2.0]  # Multiple scales for robustness
        for sigma in bandwidth_range:
            XX += sigma**2 * (sigma**2 + dxx)**(-1)  # k(x_i, x_j)
            YY += sigma**2 * (sigma**2 + dyy)**(-1)  # k(y_i, y_j)
            XY += sigma**2 * (sigma**2 + dxy)**(-1)  # k(x_i, y_j)

    elif kernel == "rbf":
        # RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||^2 / (2σ^2))
        # Multiple bandwidths make the test more sensitive to different scales
        bandwidth_range = [0.1, 0.5, 1.0, 2.0, 5.0]
        for sigma in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / (sigma**2))  # k(x_i, x_j)
            YY += torch.exp(-0.5 * dyy / (sigma**2))  # k(y_i, y_j) 
            XY += torch.exp(-0.5 * dxy / (sigma**2))  # k(x_i, y_j)

    # Step 6: Compute empirical MMD^2 estimate
    # MMD^2 ≈ (1/n^2) * sum_ij k(x_i, x_j) + (1/m^2) * sum_ij k(y_i, y_j) - (2/nm) * sum_ij k(x_i, y_j)
    # We use torch.mean which computes the average, effectively doing the normalization
    return torch.mean(XX + YY - 2. * XY)

def test_mmd_performance():
    """
    Test MMD with known distributions to verify implementation.
    This function demonstrates how MMD behaves with:
    1. Same distributions (should be close to 0)
    2. Different distributions (should be > 0)
    3. Different variances (should detect the difference)
    """
    print("Testing MMD implementation...")

    # Test 0: Same distribution - with the same samples
    torch.manual_seed(42)
    t1 = torch.randn(1000, 20) # N(0, 1)
    
    mmd_same_samples = MMD(t1, t1, 'multiscale')
    print(f"MMD between the exact same sumples: {mmd_same_samples: .6f} (should be 0)")
    
    # Test 1: Same distribution - should give low MMD
    torch.manual_seed(42)
    x1 = torch.randn(1000, 5)  # N(0, 1)
    x2 = torch.randn(1000, 5)  # N(0, 1)
    
    mmd_same = MMD(x1, x2, "multiscale")
    print(f"MMD between two N(0,1) samples: {mmd_same:.6f} (should be close to 0)")
    
    # Test 2: Different means - should give higher MMD
    y1 = torch.randn(1000, 5)           # N(0, 1)
    y2 = torch.randn(1000, 5) + 2.0     # N(2, 1)
    
    mmd_diff_mean = MMD(y1, y2, "multiscale")
    print(f"MMD between N(0,1) and N(2,1): {mmd_diff_mean:.6f} (should be > 0)")
    
    # Test 3: Different variances - should give higher MMD
    z1 = torch.randn(1000, 5)           # N(0, 1)
    z2 = torch.randn(1000, 5) * 2.0     # N(0, 4)
    
    mmd_diff_var = MMD(z1, z2, "multiscale")
    print(f"MMD between N(0,1) and N(0,4): {mmd_diff_var:.6f} (should be > 0)")
    
    # Test 4: Compare kernels
    mmd_multiscale = MMD(z1, z2, "rbf")
    print(f"MMD with rbf kernel: {mmd_multiscale:.6f}")
    
    print("Test completed!")


def compute_mmd_to_standard_normal(samples, kernel="rbf"):
    """
    Compute MMD between samples and standard normal distribution.
    
    Args:
        samples: tensor of shape [n_samples, n_features]
        kernel: "rbf" or "multiscale"
    
    Returns:
        MMD value between samples and N(0, 1)
    """
    n_samples, n_features = samples.shape
    
    # Generate samples from standard normal
    standard_normal = torch.randn(n_samples, n_features).to(samples.device)
    
    return MMD(samples, standard_normal, kernel)

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_mmd_performance()
