import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "./ai_image_generator_module/SavedModels/sd3-medium",
    torch_dtype=torch.float16,
)

pipe.enable_attention_slicing("max")         
pipe.enable_sequential_cpu_offload()        

image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=20,    
    guidance_scale=7.0,
).images[0]

image.save("./Images/cat_sd3_optimized.png")
