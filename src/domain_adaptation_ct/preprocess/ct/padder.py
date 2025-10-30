from typing import Tuple, Optional
import numpy as np

class Padder:
  """Generic 2D image padder."""
  def __init__(
    self, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, 
    arr_shape: Optional[Tuple[int, int]] = None,
    circumscribe: bool = False,
  ):
    """
    Set up a padder, which can be used to pad and then unpad images by a
    consistent amount.

    Args:
      arr_shape (optional): The shape of the image that will be padded.
      circumscribe (optional): If this is True, then an extra padding will
        be added in all directions which ensures that the padded image
        fully contains a circle which circumscribes an array of shape
        `arr_shape`. This makes the padded image safe to use with the
        Radon transform, which expects no information to be lost when it
        rotates the array.
    """
    assert pad_top >= 0
    assert pad_bottom >= 0
    assert pad_left >= 0
    assert pad_right >= 0

    self.pad_top = pad_top
    self.pad_bottom = pad_bottom
    self.pad_left = pad_left
    self.pad_right = pad_right
    self.arr_shape = arr_shape

    self.arr_padded_shape = None
    if self.arr_shape is not None:
      self.arr_padded_shape = (
        self.pad_top + self.pad_bottom + self.arr_shape[0],
        self.pad_left + self.pad_right + self.arr_shape[1],
      )

    if circumscribe:
      assert self.arr_shape is not None, "I need to know the shape of the arrays I am going to pad in order to know how to circumscribe the arrays."

      max_dimension = max(self.arr_padded_shape)
      diagonal = np.sqrt(2) * max_dimension
      circumscribing_pad = int(np.ceil((diagonal - max_dimension) / 2))
  
      self.pad_top += circumscribing_pad
      self.pad_bottom += circumscribing_pad
      self.pad_left += circumscribing_pad
      self.pad_right += circumscribing_pad

      self.arr_padded_shape = (
        self.pad_top + self.pad_bottom + self.arr_shape[0],
        self.pad_left + self.pad_right + self.arr_shape[1],
      )

  def pad(self, arr: np.ndarray):
    return np.pad(
      arr, 
      pad_width=(
        (self.pad_top, self.pad_bottom),
        (self.pad_left, self.pad_right)
      ), 
      mode="constant",
      constant_values=0
    )

  def unpad(self, arr: np.ndarray):
    return arr[
      self.pad_top:(arr.shape[0]-self.pad_bottom),
      self.pad_left:(arr.shape[1]-self.pad_right)
    ]

  @property
  def pad_amounts(self):
    return self.pad_top, self.pad_bottom, self.pad_left, self.pad_right

class SymmetricPadder(Padder):
  """Convenience wrapper for Padder to pad equally in all directions."""
  def __init__(
    self, pad_sz: int,
    arr_shape: Optional[Tuple[int, int]] = None,
    circumscribe: bool = False,
  ):
    super().__init__(pad_sz, pad_sz, pad_sz, pad_sz, arr_shape, circumscribe)

class ShiftPadder(Padder):
  """
  Pad an image to emulate shifting the isocenter of CT reconstruction.
  """
  def __init__(
    self, shift_down: int, shift_right: int,
    arr_shape: Optional[Tuple[int, int]] = None,
    circumscribe: bool = False,
  ):
    """
    Args:
      shift_down: A number of pixels by which to shift the image in the vertical direction.
        To shift the image downward, specify a positive shift_down.
        To shift the image upward, specify a negative shift_down.
      shift_right: A number of pixels by which to shift the image in the horizontal direction.
        To shift the image to the right, specify a positive shift_right.
        To shift the image to the left, specify a negative shift_right.
    """
    # Retain a square image.
    shift_vertical = abs(shift_down)
    shift_horizontal = abs(shift_right)
    shift_max = max(shift_horizontal, shift_vertical)

    pad_top = shift_max + shift_down
    pad_bottom = shift_max - shift_down
    pad_left = shift_max + shift_right
    pad_right = shift_max - shift_right

    super().__init__(pad_top, pad_bottom, pad_left, pad_right, arr_shape, circumscribe)