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
    response = OllamaChatService.change_color(user_input, colors_list, think = think)
    return {
        "colors": colors_list,
        "color_to_change": response
    }

@router.post("/chat/")
def chat(prompt: str, model: str = "deepseek-r1:8b"):
    response = OllamaChatService.chat(prompt, model)
    return {"response": response}
