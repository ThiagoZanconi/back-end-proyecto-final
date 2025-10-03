import os
from dotenv import load_dotenv
import torch
from diffusers import FluxPipeline

# Load environment variables
load_dotenv()

# ------------------------------
# Settings
# ------------------------------
class Config:
    torch_dtype: str = "float16"
    low_cpu_mem_usage: bool = False
    cpu_offload: bool = "sequential"
    num_inference_steps: int = 4
    image_height: int = 512
    image_width: int = 512
    max_sequence_length: int = 512

config = Config()

# ------------------------------
# Flux Model Base Class
# ------------------------------
class FluxModel:
    def __init__(self, name: str, save_path: str, model: str) -> None:
        self.name = name
        self.save_path = save_path
        self.model = model
        self.pipeline = None  # Initialize empty

    def load_model(self):
        if self.pipeline is None:  # Load only once
            if config.torch_dtype == "float16":
                self.pipeline = FluxPipeline.from_pretrained(
                    self.save_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=config.low_cpu_mem_usage,
                )
            self.pipeline.enable_sequential_cpu_offload()
            self.pipeline.enable_attention_slicing("max")
            self.pipeline.enable_vae_slicing()
        return self.pipeline

    def get_image(self, prompt: str, file_name: str):
        pipeline = self.load_model()
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
        image.save(f"{file_name}.png")

# ------------------------------
# Specific Model
# ------------------------------
class Flux1Schnell(FluxModel):
    def __init__(self) -> None:
        name = "black-forest-labs/FLUX.1-schnell"
        save_path = "./ai_image_generator_module/SavedModels/Schnell"
        model = "Schnell"
        super().__init__(name, save_path, model)

# ------------------------------
# Run example if script is executed
# ------------------------------
if __name__ == "__main__":
    schnell = Flux1Schnell()
    schnell.get_image(
        "A yellow squirrlel in a magical forest fighting against a bear. Both are wearing medieval armor and holding swords.",
        "fight4",
    )
