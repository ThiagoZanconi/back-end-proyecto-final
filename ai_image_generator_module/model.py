import os
from dotenv import load_dotenv

load_dotenv()

import torch

from huggingface_hub import login

login(token=os.getenv("API_KEY"))

from diffusers import FluxPipeline

# Settings
class Config:
    torch_dtype: str = os.getenv("TORCH_DTYPE")
    low_cpu_mem_usage: bool = os.getenv("LOW_CPU_MEM_USAGE")
    cpu_offload: bool = os.getenv("CPU_OFFLOAD")
    num_inference_steps: int = int(os.getenv("NUM_INFERENCE_STEPS"))
    image_height: int = int(os.getenv("IMAGE_HEIGHT"))
    image_width: int = int(os.getenv("IMAGE_WIDTH"))
    max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH"))

config = Config()

class FluxModel:
    def __init__(self, name: str, save_path: str, model: str) -> None:
        self.name = name
        self.save_path = save_path
        self.model = model
        self.pipeline = None  # 🔹 se inicializa vacío

    def load_model(self):
        if self.pipeline is None:  # 🔹 se carga solo una vez
            if(config.torch_dtype == "float16"):
                self.pipeline = FluxPipeline.from_pretrained(
                    self.save_path,   # usás el modelo ya guardado
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=config.low_cpu_mem_usage,
                )
            if(config.torch_dtype == "bfloat16"):
                self.pipeline = FluxPipeline.from_pretrained(
                    self.save_path,   # usás el modelo ya guardado
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=config.low_cpu_mem_usage,
                )
            if config.cpu_offload == "sequential":
                self.pipeline.enable_sequential_cpu_offload()               # Usa mucha memoria RAM y algo de GPU
            if config.cpu_offload == "model":
                self.pipeline.enable_model_cpu_offload()                   # Usa menos memoria RAM y mucha GPU
            self.pipeline.enable_attention_slicing("max")
            self.pipeline.enable_vae_slicing()
        return self.pipeline

    def get_image(self, prompt: str, file_name: str):
        pipeline = self.load_model()   # 🔹 reutiliza el mismo
        image = pipeline(
            prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=config.num_inference_steps,
            height=config.image_height,
            width=config.image_width,
            max_sequence_length=config.max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        image.save(f"./Images/{file_name}.png")
  