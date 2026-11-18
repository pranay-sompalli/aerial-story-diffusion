import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Computes the Fréchet Distance between two multivariate Gaussians:
    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))
    
    This is the standard metric used to evaluate image generation models (like FID),
    where lower values indicate that the generated image distribution is closer to
    the real image distribution in a feature space (e.g., Inception feature space).
    """
    # Ensure means are 1D arrays and covariances are 2D matrices
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Sanity checks to ensure both distributions have matching dimensions
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    # Compute difference between the mean vectors
    diff = mu1 - mu2

    # Calculate the square root of the product of the covariances (sqrtm)
    # The product might be nearly singular or ill-conditioned, so we handle that case.
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print('fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical precision issues can sometimes yield complex numbers with tiny imaginary parts.
    # We check if they are negligibly small and discard them, keeping only the real part.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    # Final calculation combining mean distance, traces of covariances, and cross-covariance trace
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def calculate_fid_given_features(feature1, feature2):
    """
    Calculates FID between two sets of activation features (e.g. from an Inception model).
    This function computes the sample means and covariance matrices for both
    feature matrices and then evaluates the Frechet Distance.
    """
    # Compute mean across the batch dimension (axis 0) for both real and generated features
    mu1 = np.mean(feature1, axis=0)
    # Compute covariance. rowvar=False means columns are variables, rows are observations
    sigma1 = np.cov(feature1, rowvar=False)
    
    mu2 = np.mean(feature2, axis=0)
    sigma2 = np.cov(feature2, rowvar=False)
    
    # Calculate Frechet Distance based on computed statistics
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid_value

