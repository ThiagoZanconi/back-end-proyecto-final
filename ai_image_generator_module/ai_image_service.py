from diffusers import AutoPipelineForText2Image
from diffusers import FluxPipeline
from ai_image_generator_module.model import Flux1Schnell

import torch

class AIImageService:
    @staticmethod
    def sdxl_text_to_image(prompt: str, path: str, width: int = 512, height: int = 512, steps: int = 4, guidance_scale: float = 7.0):
        assert width % 8 == 0 and height % 8 == 0, "Width y height deben ser múltiplos de 8"

        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo",variant="fp16")
        pipe.to("cuda")

        print(f"Generating image... Width: {width}, Height: {height}, Steps: {steps}, Guidance Scale: {guidance_scale}")
        #Guidance scale:
        #0.0 → sin guía, salida muy libre/caótica.
        #1.0–3.0 → baja influencia del prompt.
        #7.0–8.0 → valor común, buen balance entre fidelidad y creatividad.
        #>10 → puede dar imágenes muy forzadas, a veces irreales o con artefactos.
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]

        image.save(path)


    @staticmethod
    def flux_text_to_image(prompt: str, path: str, width: int = 512, height: int = 512, steps: int = 4):
        schnell = Flux1Schnell()
        schnell.get_image(prompt, path, width, height, steps)

        
