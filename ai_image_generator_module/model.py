import torch
from diffusers import FluxPipeline
from pathlib import Path

# ------------------------------
# Flux Model Base Class
# ------------------------------
class FluxModel:
    def __init__(self, name: str, save_path: str, model: str) -> None:
        self.name = name
        self.save_path = save_path
        self.model = model
        self.pipeline = None 

    def load_model(self):
        if self.pipeline is None: 
            self.pipeline = FluxPipeline.from_pretrained(
            self.name,
            torch_dtype=torch.float16,  
            low_cpu_mem_usage=False,
        )

            self.pipeline.enable_sequential_cpu_offload()
            self.pipeline.enable_attention_slicing("max")
            self.pipeline.enable_vae_slicing()
        return self.pipeline

    def get_image(self, prompt: str, path: str, width: int = 512, height: int = 512, steps: int = 4, guidance_scale: float = 0.0):
        pipeline = self.load_model()
        image = pipeline(
            prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=steps,
            height=height,
            width=width,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        save_path = Path(path).with_suffix(".png")
        image.save(save_path)

class Flux1Schnell(FluxModel):
    def __init__(self) -> None:
        name = "black-forest-labs/FLUX.1-schnell"
        save_path = "./ai_image_generator_module/SavedModels/Schnell"
        model = "Schnell"
        super().__init__(name, save_path, model)

if __name__ == "__main__":
    schnell = Flux1Schnell()
    schnell.get_image(
        "A purple squirrlel in a magical forest fighting against a bear. Both are wearing medieval armor and holding swords.",
        "Images/fight5",
    )
