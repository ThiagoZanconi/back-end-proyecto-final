from typing import List
from fastapi import APIRouter
import numpy as np
from ai_assistant_module.ollama_chat_service import OllamaChatService
from image_processing_module.image_processing_service import ImageProcessingService

router = APIRouter(prefix="/ai_assistant", tags=["AI Assistant"])

@router.post("/change_colors/")
def change_colors(path: str, user_input: str, n: int = 10, delta_threshold: float = 3.0, think: bool = False):
    from main import get_image_service
    image_processing_service: ImageProcessingService = get_image_service()
    colors: List[np.ndarray] = image_processing_service.get_main_different_colors_rgb(path, n, delta_threshold)
    colors_list = []
    for c in colors:
        colors_list.append([int(c[0]), int(c[1]), int(c[2])])
    response = OllamaChatService.change_item_color(user_input, colors_list, think = think)
    return {
        "colors": colors_list,
        "color_to_change": response
    }

@router.post("/chat/")
def chat(prompt: str, model: str = "deepseek-r1:8b"):
    response = OllamaChatService.chat(prompt, model)
    return {"response": response}

@router.post("/generate_image/")
def generate_image(prompt:str, image_width: int = 512, image_height: int = 512, image_steps: int = 4, guidance_scale: float = 7.0):
    from main import file_path
    from ai_image_generator_module.sdxl_turbo import SDXLTurbo
    path = file_path / "generated_image.png"
    SDXLTurbo.text_to_image(prompt, path, width=image_width, height=image_height, steps=image_steps, guidance_scale=guidance_scale)
    return {"image_path": path}

@router.post("/perform_action/")
def perform_action(path: str, user_input: str, n: int = 10, delta_threshold: float = 3.0, think: bool = False, model: str = "deepseek-r1:8b", image_width: int = 512,
                image_height: int = 512, image_steps: int = 4, guidance_scale: float = 7.0
                   ):
    action = OllamaChatService.select_action(user_input, think = think)
    if "1" in action:
        return __change_item_color(path, user_input, n, delta_threshold, think)
    elif "2" in action:
        return __change_background_color(path)
    elif "3" in action:
        return generate_image(user_input, image_width=image_width, image_height=image_height, image_steps=image_steps, guidance_scale=guidance_scale)
    elif "4" in action:
        return chat(user_input, model)
    else:
        return {"response": "Invalid user input. Please try again."}

def __change_item_color(path: str, user_input: str, n: int = 10, delta_threshold: float = 3.0, think: bool = False):
    from main import get_image_service
    image_processing_service: ImageProcessingService = get_image_service()
    colors: List[np.ndarray] = image_processing_service.get_main_different_colors_rgb(path, n, delta_threshold)
    colors_list = []
    for c in colors:
        colors_list.append([int(c[0]), int(c[1]), int(c[2])])
    item = OllamaChatService.get_item(user_input, think = think)
    item_color = OllamaChatService.get_item_color(item, colors_list, think = think)
    color = OllamaChatService.get_new_color(user_input, think = think)
    return {
        "colors": colors_list,
        "item": item,
        "item_color": item_color,
        "new_color": color
    }

def __change_background_color(path: str):
    from main import get_image_service
    image_processing_service: ImageProcessingService = get_image_service()
    #image_processing_service.remove_background(path)
    return {
        "msg": "Background removed successfully"
    }
    