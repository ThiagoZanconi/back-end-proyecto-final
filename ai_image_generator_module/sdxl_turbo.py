from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A pixelart image of a skeleton with an ice sword."
image = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0).images[0]

image.save("./Images/Ice-Skeleton.png")
