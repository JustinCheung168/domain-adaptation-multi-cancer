from typing import List, Optional

# Image manipulation
import numpy as np

class SinogramCorruptor:
  """
  Adds an artifact in the sinogram.
  """
  def create_multiplicative_ring_artifact(self, sinogram: np.ndarray, detectors: List[int], factors: List[float]):
    """
    Create a ring artifact characterized by a detector only receiving a certain
    fraction of incoming X-rays.

    For example, setting `detectors` to [5, 7] and `factors` to [0.2, 0.5]
    causes detector at index 5 to receive only 20% of the incoming X-rays,
    and detector at index 7 to receive only 50% of the incoming X-rays.

    Args:
      sinogram: A 2D array of shape (n_detectors, n_angles) representing the
        sinogram.
      detectors: A list of detector indices that will receive the artifact.
      factors: A list of multiplicative factors by which to multiply each
        detector. One element in `factors` corresponds to one index in
        `detectors`.

    Returns:
      A 2D array of shape (n_detectors, n_angles) representing the corrupted sinogram.
    """
    assert len(detectors) == len(factors), "Must provide a factor for each detector"

    corrupted_sinogram = sinogram.copy()
    for i in range(len(detectors)):
      corrupted_sinogram[detectors[i], :] = corrupted_sinogram[detectors[i], :] * factors[i]

    return corrupted_sinogram

  def create_multiplicative_streak_artifact(self, sinogram: np.ndarray, views: List[int], factors: Optional[List[float]] = None):
    """
    Create a ring artifact characterized by a view (projection) being attenuated by a certain factor.

    For example, setting `views` to [5, 7] and `factors` to [0.2, 0.5]
    causes the view at index 5 to be attenuated to 20% of its original intensity,
    and the view at index 7 to be attenuated to 50% of its original intensity.

    It is likely not realistic for a view to be attenuated partially;
    more likely it would be missing entirely, which can be implemented by excluding `factors`.

    Args:
      sinogram: A 2D array of shape (n_detectors, n_angles) representing the
        sinogram.
      views: A list of view indices that will receive the artifact.
      factors: A list of multiplicative factors by which to multiply each
        view. One element in `factors` corresponds to one index in
        `views`.

    Returns:
      A 2D array of shape (n_detectors, n_angles) representing the corrupted sinogram.
    """
    if factors is None:
      # Assume the views are completely missing.
      factors = [0.0] * len(views)
    else:
      assert len(views) == len(factors), "Must provide a factor for each detector"

    corrupted_sinogram = sinogram.copy()
    for i in range(len(views)):
      corrupted_sinogram[:, views[i]] = corrupted_sinogram[:, views[i]] * factors[i]

    return corrupted_sinogram