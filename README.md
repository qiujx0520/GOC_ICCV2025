# GOC_ICCV2025
## Accelerating Diffusion Transformer via Gradient-Optimized Cache


### [Paper](https://arxiv.org/abs/2503.05156)

## Requirement
With pytorch(>2.0) installed, execute the following command to install necessary packages
```
pip install accelerate diffusers timm torchvision wandb
```
## Checkpoint

[DiT-XL-2-256x256.pt](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)

```
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
```

## Prior Knowledge Extraction
Change the cache matrix in ~/GOC/DiT/models/models.py to matrix1
```
python 1-blocks.py
python 2-block_pth_average.py
```

## Sample Image
For learning-to-cache DDIM-20 + GOC:
```
python sample.py --model DiT-XL/2 --num-sampling-steps 20 --ddim-sample --accelerate-method dynamiclayer --path ckpt/DDIM20_router.pt --thres 0.1
```

For FORA + GOC,
Change the cache matrix in ~/GOC/DiT/models/models.py to matrix3:
```
python sample.py --model DiT-XL/2 --num-sampling-steps 20 --ddim-sample 
```

## Sample 50k images for Evaluation
If you want to reproduce the FID results from the paper, you can use the following command to sample 50k images:

For learning-to-cache DDIM-20 + GOC:
```
torchrun --nnodes=1 --nproc_per_node=8 --master_port 12345 sample_ddp.py --model DiT-XL/2 --num-sampling-steps NUM_STEPS --ddim-sample --accelerate-method dynamiclayer --path PATH_TO_TRAINED_ROUTER --thres 0.1
```
For FORA + GOC:
```
torchrun --nnodes=1 --nproc_per_node=8 --master_port 12345 sample_ddp.py --model DiT-XL/2 --num-sampling-steps NUM_STEPS --ddim-sample --path PATH_TO_TRAINED_ROUTER --thres 0.1
```
Be sure to modify NUM_STEPS and PATH_TO_TRAINED_ROUTER to correspond to the respective NFE steps and the location of the router.

## Calculate FID
We follow DiT to evaluate FID by [the code](https://github.com/openai/guided-diffusion/tree/main/evaluations). Please install the required packages, download the pre-computed sample batches, and then run the following command:
```
python evaluator.py ~/ckpt/VIRTUAL_imagenet256_labeled.npz PATH_TO_NPZ
```

## BibTeX

```bibtex
@article{qiu2025accelerating,
  title={Accelerating diffusion transformer via gradient-optimized cache},
  author={Qiu, Junxiang and Liu, Lin and Wang, Shuo and Lu, Jinda and Chen, Kezhou and Hao, Yanbin},
  journal={arXiv preprint arXiv:2503.05156},
  year={2025}
}
```

## Acknowledgement
This implementation is based on [DiT](https://github.com/facebookresearch/DiT) and [Learning-to-cache](https://github.com/horseee/learning-to-cache). 
