from typing import List
from fastapi import APIRouter
import uuid
import numpy as np
from ai_assistant_module.ollama_chat_service import OllamaChatService
from image_processing_module.image_processing_service import ImageProcessingService
import re

router = APIRouter(prefix="/ai_assistant", tags=["AI Assistant"])

@router.post("/change_colors/")
def change_color(filename: str, user_input: str, n: int = 10, delta_threshold: float = 3.0, think: bool = False):
    from main import get_image_service
    image_processing_service: ImageProcessingService = get_image_service()
    colors: List[np.ndarray] = image_processing_service.get_main_different_colors_rgb(filename, n, delta_threshold)
    colors_list = []
    for c in colors:
        colors_list.append([int(c[0]), int(c[1]), int(c[2])])
    first_color = OllamaChatService.change_item_color(user_input, colors_list, think = think)
    second_color = OllamaChatService.get_second_color(user_input, think = think)
    print(f"First color: {first_color}, Second color: {second_color}")

    first_color_List = list(map(int, re.findall(r"-?\d+", first_color)))
    second_color_List = list(map(int, re.findall(r"-?\d+", second_color)))

    new_filename = image_processing_service.change_gamma_colors(filename, first_color_List, second_color_List, delta_threshold)
    return {
        "msg": "Gamma colors changed correctly",
        "filename": new_filename
    }

@router.post("/chat/")
def chat(prompt: str, model: str = "deepseek-r1:8b", think = False, temperature:float = 0.2, top_k:float = 8.0, top_p: float = 0.4):
    response = OllamaChatService.chat(prompt, model, think, temperature, top_k, top_p)
    return {"response": response}

@router.post("/generate_image/")
def generate_image(prompt:str, image_width: int = 512, image_height: int = 512, image_steps: int = 4, guidance_scale: float = 7.0, ai_image_model: str = "sdxl-turbo"):
    from main import file_path
    from ai_image_generator_module.ai_image_service import AIImageService
    filename = f"{uuid.uuid4()}.png"
    path = file_path / filename
    if ai_image_model == "sdxl-turbo":
        AIImageService.sdxl_text_to_image(prompt, path, width=image_width, height=image_height, steps=image_steps, guidance_scale=guidance_scale)
    if ai_image_model == "flux-1-schnell":
        AIImageService.flux_text_to_image(prompt, path, width=image_width, height=image_height, steps=image_steps)
    if ai_image_model == "gemini-2.5-flash-image":
        AIImageService.gemini_text_to_image(prompt, path)
    return {"filename": filename}

@router.post("/perform_action/")
def perform_action(filename: str, user_input: str, n: int = 10, delta_threshold: float = 3.0, think: bool = False, ai_text_model: str = "deepseek-r1:8b",
        ai_image_model: str = "sdxl-turbo", image_width: int = 512, image_height: int = 512, image_steps: int = 4, guidance_scale: float = 7.0, 
        temperature:float = 0.2, top_k:float = 8.0, top_p: float = 0.4):
    
    action = OllamaChatService.select_action(user_input, think = think, temperature=temperature, top_k=top_k, top_p=top_p)
    if "1" in action:
        return change_color(filename, user_input, n, delta_threshold, think)
    elif "2" in action:
        return __change_background_color(filename)
    elif "3" in action:
        return generate_image(user_input, image_width=image_width, image_height=image_height, image_steps=image_steps, guidance_scale=guidance_scale, ai_image_model=ai_image_model)
    elif "4" in action:
        return chat(user_input, ai_text_model, temperature=temperature, top_k=top_k, top_p=top_p)
    else:
        return {"response": "Invalid user input. Please try again."}

def __change_background_color(path: str):
    from main import get_image_service
    image_processing_service: ImageProcessingService = get_image_service()
    #image_processing_service.remove_background(path)
    return {
        "msg": "Background removed successfully"
    }
    