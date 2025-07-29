# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
# Allow TF32 for CUDA matrix multiplication
torch.backends.cuda.matmul.allow_tf32 = True
# Allow TF32 for cuDNN operations
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

import argparse
import numpy as np
import os
# Set the visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main(args):
    # Setup PyTorch:
    # Disable gradient calculation
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Initialize the diffusion process
    diffusion = create_diffusion(str(args.num_sampling_steps))  

    # Load model:
    latent_size = args.image_size // 8
    if args.accelerate_method is not None and args.accelerate_method == "dynamiclayer":
        from models.dynamic_models import DiT_models
    else:
        from models.models import DiT_models

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    if args.accelerate_method is not None and 'dynamiclayer' in args.accelerate_method:
        model.load_ranking(args.path, args.num_sampling_steps, diffusion.timestep_map, args.thres)
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    # Set the model to evaluation mode
    model.eval()  
    vae = AutoencoderKL.from_pretrained(f"~/GOC/DiT/pretrained_models/sd-vae-ft-ema").to(device)

    for i in range (1000):
        # Set the random seed
        torch.manual_seed(args.seed)
        # Labels to condition the model with (feel free to change):
        label = i
        class_labels = [label]

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Used to collect the output of each block
        block_outputs = {}

        # Define the hook function to directly store all outputs
        def hook_fn(module, input, output, block_idx):
            # print(f"Block {block_idx} output type: {type(output)}, content: {output}")
            if block_idx not in block_outputs:
                block_outputs[block_idx] = []  # Initialize a list to store the output of each step
            block_outputs[block_idx].append(output)  # Append the output of each step

        # Register the hook to each block
        hooks = []
        for block_idx, block in enumerate(model.blocks):  # Assume the model has a 'blocks' attribute
            hook = block.register_forward_hook(lambda module, input, output, idx=block_idx: hook_fn(module, input, output, idx))
            hooks.append(hook)


        _ = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

        # remove hook
        for hook in hooks:
            hook.remove()

        # save block_outputs 
        save_path = f'~/GOC/learning-to-cache-main/DiT/block_pth/{label}_block_outputs.pth'
        torch.save(block_outputs, save_path)

        print("every block_outputs have been saved:", save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--accelerate-method", type=str, default=None,
                        help="Use the accelerated version of the model.")
    
    parser.add_argument("--ddim-sample", action="store_true", default=False,)
    parser.add_argument("--p-sample", action="store_true", default=False,)
    
    parser.add_argument("--path", type=str, default=None,
                        help="Optional path to a router checkpoint")
    parser.add_argument("--thres", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
