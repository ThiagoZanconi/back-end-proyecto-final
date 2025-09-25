from diffusers import AutoPipelineForText2Image
import torch

class SDXLTurbo:
    @staticmethod  
    def text_to_image(prompt: str, path: str):
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16, variant="fp16")
        pipe.to("cuda")
        image = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        image.save(path)
