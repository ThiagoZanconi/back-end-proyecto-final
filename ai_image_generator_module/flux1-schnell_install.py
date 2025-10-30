import os
from dotenv import load_dotenv

load_dotenv()

import torch

from huggingface_hub import login

login(token=os.getenv("API_KEY"))


from diffusers import AutoPipelineForText2Image

torch_dtype=os.getenv("TORCH_DTYPE")

pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype)
pipeline.save_pretrained("./SavedModels/sdxl-turbo")


