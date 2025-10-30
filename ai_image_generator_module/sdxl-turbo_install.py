import os
from dotenv import load_dotenv

load_dotenv()

import torch

from huggingface_hub import login

login(token=os.getenv("API_KEY"))


from diffusers import AutoPipelineForText2Image

# "black-forest-labs/FLUX.1-schnell"

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16)
pipeline.save_pretrained("./SavedModels/sdxl-turbo")


