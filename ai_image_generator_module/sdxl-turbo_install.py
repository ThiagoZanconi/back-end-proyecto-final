import os
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import login

login(token=os.getenv("API_KEY"))

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
pipeline.save_pretrained("./SavedModels/sdxl-turbo")


