from diffusers import FluxPipeline
from PIL import Image
import torch
import os

def main():
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    model_dir = os.path.join("SavedModels", "FLUX.1-Kontext-dev")
    os.makedirs(model_dir, exist_ok=True)

    input_image_path = "ai_image_generator_module/input.png"  
    prompt = "turn the photo into a watercolor painting"
    output_image_path = "edited_image_flux.png"

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"{input_image_path} not found.")
    init_image = Image.open(input_image_path).convert("RGB")

    print("Loading Flux Kontext model (float16)...")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=model_dir
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    if device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3:
        pipe.enable_model_cpu_offload()

    print("Editing image with Flux Kontext...")
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=0.7,            
        guidance_scale=7.5,
        num_inference_steps=20
    ).images[0]

    result.save(output_image_path)
    print(f"Saved edited image to {output_image_path}")

if __name__ == "__main__":
    main()
