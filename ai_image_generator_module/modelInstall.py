import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "./ai_image_generator_module/SavedModels/sd15",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=5,
    guidance_scale=7.0,
).images[0]

image.save("./Images/cat_sd15.png")