from diffusers import AutoPipelineForText2Image
from diffusers import FluxPipeline
from ai_image_generator_module.model import Flux1Schnell
import os
from google import genai
from PIL import Image
from io import BytesIO

import torch

class AIImageService:
    @staticmethod
    def sdxl_text_to_image(prompt: str, path: str, width: int = 512, height: int = 512, steps: int = 4, guidance_scale: float = 7.0):
        assert width % 8 == 0 and height % 8 == 0, "Width y height deben ser múltiplos de 8"

        pipe = AutoPipelineForText2Image.from_pretrained("./ai_image_generator_module/SavedModels/sdxl-turbo",variant="fp16")
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

    @staticmethod
    def gemini_text_to_image(prompt: str, path: str):
        api_key = os.getenv("DEEPMIND_API_KEY")
        if not api_key:
            raise RuntimeError("Definí la variable DEEPMIND_API_KEY con tu API Key")
        client = genai.Client(api_key=api_key)
        # Configuración para que la respuesta incluya imagen (modo multimodal)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
            # indicamos que queremos imagen en la respuesta
            config=genai.types.GenerateContentConfig(
                response_modalities=["Image"]
            )
        )
        # Extraer la imagen de la respuesta
        # Asumimos que la primera “candidate” tiene la imagen
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if part.inline_data is not None:
                img = Image.open(BytesIO(part.inline_data.data))
                img.save(path)
                return
        
        raise RuntimeError("No se encontró parte de imagen en la respuesta")
    

        
