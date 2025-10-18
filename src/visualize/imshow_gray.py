"""Custom matplotlib wrapper for this application."""

from typing import Optional, Tuple, List
from matplotlib.image import AxesImage

# Image display
import matplotlib.pyplot as plt

# Image manipulation
import numpy as np

class ImShowGray:
  @staticmethod
  def imshow(img: np.ndarray, 
            axes: Optional[plt.Axes] = None,
            title: Optional[str] = None,
            window: Optional[Tuple[float, float]] = None,
            title_stats: bool = False,
            title_window: bool = False
  ) -> plt.Axes:
    """
    Display grayscale image with preferred settings.

    Window is specified as (vmin, vmax) - so the minimum and maximum intensity values.
    Values under vmin will be visualized as black,
    whereas values above vmax will be visualized as white.
    Recommend window=(0, 255) for 8-bit images.
    """
    if (axes is None):
      _, axes = plt.subplots(1, 1)
    
    if window is None:
      window = (np.nanmin(img).item(), np.nanmax(img).item())

    if title is None:
      title = ""
    if title_stats:
      title += "\n" + ImShowGray.stat_string(img)
    if title_window:
      title += "\n" + f"Window: [{window[0]:.2f}, {window[1]:.2f}]"
    axes.set_title(title,fontsize=19)


    axes.imshow(img, cmap='gray', vmin=window[0], vmax=window[1])
    axes.axis('off')

    return axes

  @staticmethod
  def stat_string(img: np.ndarray) -> str:
    """Produce a string with common simple statistics of interest"""
    return f"Max: {np.nanmax(img):.4f}" + \
           f"\nMin: {np.nanmin(img):.4f}" + \
           f"\nMean: {np.nanmean(img):.4f}" + \
           f"\nStdev: {np.nanstd(img):.4f}" + \
           f"\n#NaN: {np.isnan(img).sum()}"

  @staticmethod
  def imshow_diff(
    img_new: np.ndarray, img_old: np.ndarray, 
    window: Optional[Tuple[float, float]] = None,
    diff_window: Optional[Tuple[float, float]] = None,
    titles: Optional[Tuple[str, str]] = None,
    title_stats: bool = False,
    title_window: bool = False,
    include_abs_diff: bool = True,
  ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Display the difference between two images for comparative purposes.
    The difference taken is img_new minus img_old.

    It is worth noting that for the difference image `img_diff`,
    if `title_stats` is enabled, then the displayed standard deviation (ddof=0)
    is equivalent to the RMSE between `img_new` and `img_old`.
    """
    if titles is None:
      titles = ("New", "Old")

    img_diff = img_new - img_old
    img_absdiff = np.abs(img_diff)

    n_subplots = 3
    if include_abs_diff:
        n_subplots = 4
    fig, axes = plt.subplots(1, n_subplots, figsize=(12, 6))

    ImShowGray.imshow(img_new, axes[0], title=titles[0], title_stats=title_stats, title_window=title_window, window=window)
    ImShowGray.imshow(img_old, axes[1], title=titles[1], title_stats=title_stats, title_window=title_window, window=window)
    diff_title=f"{titles[0]} - {titles[1]}"
    diff_title=f"Difference"
    ImShowGray.imshow(img_diff, axes[2], title=diff_title, title_stats=title_stats, title_window=title_window, window=diff_window)
    if include_abs_diff:
        ImShowGray.imshow(img_absdiff, axes[3], title=f"|{titles[0]} - {titles[1]}|", title_stats=title_stats, title_window=title_window, window=diff_window)

    return fig, axes
