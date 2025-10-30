from enum import Enum

# Progress bar
from tqdm import tqdm

# Image manipulation
import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.ndimage import rotate

class KernelType(Enum):
  RAMP = 0
  HANN_RAMP = 1

class Projector:
  """
  Implements discrete Radon transform (forward projection)
  and its inverse (backprojection).

  For demonstration purposes only. `skimage.transform.radon` is recommended - 
  it is far more optimized, and this code has some issue with normalization
  which `skimage.transform.radon` handles correctly.

  Args:
    n_detectors: The number of simulated detectors used in forward projection.
      Because projection is done by summing across rows in this implementation,
      this must equal the number of rows in the image before forward projection.
    recon_width: The number of columns to give the result of backprojection.
      In this simulation, you likely want this to equal the number of columns
      in the image before forward projection.
    angles_deg: An array of angles (in degrees) at which to sample a given image
      during forward projection.
    kernel_type: The type of filter to apply to the sinogram before
      backprojection. For now, KernelType.HANNING_RAMP is recommended.
  """

  # ===== PUBLIC METHODS =====

  def __init__(self, n_detectors: int, recon_width: int,
               angles_deg: np.ndarray, kernel_type: KernelType):
    self.n_detectors = n_detectors
    self.recon_width = recon_width
    self.angles_deg = angles_deg
    self.n_angles = len(angles_deg)

    # The next highest power of 2 after the size of the sinogram.
    # Used to expedite FFT.
    self.n_detectors_pow2 = 2 ** int(np.ceil(np.log2(n_detectors)))

    self.kernel = self._calculate_kernel(self.n_detectors_pow2, kernel_type)

  def forward_project(self, img: np.ndarray):
    """
    Build the sinogram of `img`.

    Projections are taken across rows of `img`.

    Args:
      img: A 2D array of shape (n_detectors, _) representing the image.

    Returns:
      A 2D array of shape (n_angles, n_detectors) representing the sinogram.
    """
    assert img.shape[0] == self.n_detectors, "Image must have same number of rows as the initialized number of detectors"

    sinogram = np.zeros((self.n_angles, self.n_detectors))

    for i_angle in tqdm(range(self.n_angles), desc="Forward Projection"):
      # Rotate the image CCW by the i-th angle.
      angle_deg = self.angles_deg[i_angle]
      img_rotated = rotate(img, angle_deg, reshape=False)

      # Project across rows.
      # Each summation approximates integration along the X-ray beam's path
      # with step size dl.
      proj = np.sum(img_rotated, axis=1)

      # Save projection as a row of the sinogram.
      sinogram[i_angle, :] = proj

    return sinogram

  def filtered_backproject(self, sinogram: np.ndarray):
    """
    Reconstruct the image from the `sinogram`.

    Args:
      sinogram: A 2D array of shape (n_angles, n_detectors) representing the
        sinogram.

    Returns:
      A 2D array of shape (n_detectors, recon_width) representing the
      reconstructed image.
    """
    return self._backproject(self._filter_sinogram(sinogram))

  # ===== PRIVATE METHODS =====

  def _calculate_kernel(self, n: int, kernel_type: KernelType):
    """
    Derive the appropriate kernel to filter the sinogram with prior to
    backprojection.

    Most kernels are a variant of a ramp function, which combats the effect of
    the oversampling of the image center during backpropagation.
    This ramp function is usually apodized to avoid amplifying
    high-frequency noise.

    Args:
      n: Kernel size. Assumed to be a power of 2.

    Returns:
      A 1D array of shape (n,) representing the kernel. This can be applied
      by elementwise multiplication in the discrete Fourier domain.
    """
    # All kernels are some function of normalized discrete frequency f,
    # covering domain [-1/2 ... 1/2)
    f = fftshift(fftfreq(n))

    if kernel_type.value == KernelType.RAMP.value:
      kernel = np.abs(f)
    elif kernel_type.value == KernelType.HANN_RAMP.value:
      kernel = np.abs(f)

      # Apodize the ramp using a Hann window.
      # The prepended 0 corresponds to the highest frequency.
      apodizer = np.concatenate(([0], np.hanning(n-1)))

      kernel = kernel * apodizer
    else:
      raise ValueError(f"Unknown apodization type: {kernel_type}")

    # Normalize the kernel to sum to 1 so that on average,
    # a filtered signal's magnitude is unmodified.
    kernel = kernel / np.sum(kernel)

    # Circularly shift the kernel so that discrete frequencies are nonnegative.
    kernel = ifftshift(kernel)

    return kernel

  def _filter_sinogram(self, sinogram):
    """
    Filter each row (projection) of the sinogram with the precalculated kernel
    through multiplication in the discrete Fourier domain.

    The sinogram is padded row-wise to the next highest power of 2
    to expedite FFT.
    """
    # Perform discrete Fourier transform on each projection.
    # Central Slice Theorem provides that these are slices of the discrete Fourier domain.
    dft_slices = fft(sinogram, n=self.n_detectors_pow2, axis=1)

    # Filter the slices in discrete Fourier domain.
    dft_slices = dft_slices * self.kernel

    # Return to sinogram space.
    sinogram_filtered = ifft(dft_slices, axis=1)

    # Remove the padding applied during FFT.
    sinogram_filtered = sinogram_filtered[:,:self.n_detectors]

    return sinogram_filtered

  def _backproject(self, sinogram: np.ndarray):
    """
    Reconstruct the image from the `sinogram`.
    Does not include the filtering that combats oversampling.

    Args:
      sinogram: A 2D array of shape (n_angles, n_detectors) representing the
        sinogram.

    Returns:
      A 2D array of shape (n_detectors, recon_width) representing the
      reconstructed image.
    """
    assert sinogram.shape[0] == self.n_angles, "Sinogram must have same number of rows as the initialized number of angles"
    assert sinogram.shape[1] == self.n_detectors, "Sinogram must have same number of columns as the initialized number of detectors"

    img_recon = np.zeros((self.n_detectors, self.recon_width))

    # Assume that the spacing between angles remains constant.
    # d_angle = np.deg2rad(np.mean(np.diff(self.angles_deg)))
    d_angle = 1.0

    for i_angle in tqdm(range(self.n_angles), desc="Backprojection"):

      # Get the i-th projection from the sinogram.
      proj = sinogram[i_angle, :]

      # Replicate the projection in the rows direction to form one backprojection.
      backproj = np.tile(proj, (self.recon_width, 1)).T

      # Rotate the backprojection back to the angle it was originally taken from during forward projection.
      angle_deg = self.angles_deg[i_angle]
      backproj_unrotated = rotate(backproj, -angle_deg, reshape=False)

      # Add this backprojection as a contribution to the final image.
      img_recon = img_recon + backproj_unrotated * d_angle

    # FFT during any filtering transforms data to complex values.
    # Assume negligible imaginary component and return real component.
    return np.real(img_recon) 