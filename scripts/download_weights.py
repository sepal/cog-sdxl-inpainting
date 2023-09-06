# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
from diffusers import StableDiffusionXLInpaintPipeline

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
    usse_safetensors=True,
)

pipe.save_pretrained("./sdxl-cache", safe_serialization=True)
