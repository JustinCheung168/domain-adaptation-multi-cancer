from typing import Callable
import time
import os

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import torch

class IntegratedGradientsAttributor():
    """"""
    def __init__(self, model_forward_func: Callable[[torch.Tensor], torch.Tensor], device):
        """
        `model_forward_func` should wrap a model, just taking an image input and providing a logits output.
        `device` should be where the model was sent. The model is expected to already be on the device prior to instantiating this.
        """
        # Prepare integrated gradients algorithm.
        self.model_forward_func = model_forward_func
        self.integrated_gradients = IntegratedGradients(self.model_forward_func)
        self.device = device


    def calculate_attribution(self, img: torch.Tensor, label: int, n_steps: int):
        """
        img is a tensor in CHW format.
        n_steps is number of samples along linear interpolation from baseline image to the instance's image to perform approximated integration over.
            A reasonable choice could be 100 or 200 depending on your GPU's memory.
        """
        # Using this as shorthand for expecting CHW format
        assert img.shape[0] == 3 
        assert len(img.shape) == 3

        t = time.time()

        image_on_device = img.unsqueeze(0).to(self.device)

        # For sanity, run inference on this image as well.
        logits = self.model_forward_func(image_on_device).detach().cpu()[0]
        pred = torch.argmax(logits).item()
        print(f"Model predicted logits: {logits} (prediction would be {pred})")

        # Handle sending to device and back internally.
        attribution = self.integrated_gradients.attribute(
            image_on_device,
            target=label,
            n_steps=n_steps
        ).detach().cpu()[0] # Drop batch dimension
        
        del image_on_device

        exec_time = time.time() - t
        print(f"Attributions calculated - {n_steps} steps, {exec_time}s total ({exec_time / n_steps}s per step)")

        return attribution

    @staticmethod
    def visualize_attribution(img: torch.Tensor, attribution: torch.Tensor, ttl_prefix: str, out_dir: str):
        assert img.shape == torch.Size([3, 224, 224])
        assert img.shape == attribution.shape
        
        os.makedirs(out_dir, exist_ok=True)

        # Permute CHW -> HWC
        img_np = img.permute(1, 2, 0).detach().cpu().numpy()
        attribution_np = attribution.permute(1, 2, 0).detach().cpu().numpy()

        # Normalize image for display
        img_np_min = img_np.min()
        img_np_minmax_normalized = (img_np - img_np_min) / (img_np.max() - img_np_min)

        # Save original image
        plt.figure()
        plt.imshow(img_np_minmax_normalized)
        plt.axis('off')
        plt.title(f"{ttl_prefix}Original Image")
        plt.savefig(os.path.join(out_dir, f"{ttl_prefix}_original.png"), bbox_inches='tight')
        plt.close()

        # Save heat map attribution
        fig1, _ = viz.visualize_image_attr(
            attribution_np,
            img_np,
            method="heat_map",
            sign="all",
            show_colorbar=True,
            cmap="gray",
            title=f"{ttl_prefix}Integrated Gradients Attribution",
        )
        fig1.savefig(os.path.join(out_dir, f"{ttl_prefix}_heatmap.png"), bbox_inches='tight')
        plt.close(fig1)

        # Save blended heat map attribution
        fig2, _ = viz.visualize_image_attr(
            attribution_np,
            img_np,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            cmap="jet",
            title=f"{ttl_prefix}Integrated Gradients Attribution & Image\nBlended Heat Map",
        )
        fig2.savefig(os.path.join(out_dir, f"{ttl_prefix}_blended.png"), bbox_inches='tight')
        plt.close(fig2)
