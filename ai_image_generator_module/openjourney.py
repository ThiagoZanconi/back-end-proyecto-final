from diffusers import StableDiffusionPipeline
import torch

model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# desactivar el safety checker
pipe = pipe.to("cuda")
prompt = "skeleton warrior holding a flaming sword, pixel art, 16-bit, 32x32 sprite, retro game style, sharp edges, limited color palette, highly detailed sprite, dark fantasy atmosphere"
image = pipe(
            prompt,
            num_inference_steps= 4,
        ).images[0]
image.save("./Images/PixelSkeleton.png")
