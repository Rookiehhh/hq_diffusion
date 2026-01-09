import diffusers
import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusion3InpaintPipeline
import sys
from diffusers.utils import load_image
import numpy as np

if __name__ == "__main__":

    input_model = sys.argv[1]
    lora_path = sys.argv[2]
    pipe = StableDiffusion3Pipeline.from_single_file(input_model, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(lora_path)
    pipe = pipe.to("cuda")

    # init_image = diffusers.utils.load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
    # mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")


    result = pipe(
        "defect of crack",
        num_inference_steps=40,
        height=512,
        width=512,
        guidance_scale=4.5,
        num_images_per_prompt=4,
    ).images
    for i, img in enumerate(result):
        img.save(f"/root/autodl-tmp/tmpimgs/test_masked_blended_{i:02d}.png")
    pass