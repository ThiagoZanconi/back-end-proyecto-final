import os
from typing import List, Tuple
import google.generativeai as genai

from ai_assistant_module.prompt_service import PromptService

class GeminiChatService:
    @staticmethod
    def chat(prompt: str, temperature:float = 0.2, top_k:int = 8, top_p: float = 0.4) -> str:
        API_KEY = os.getenv("GEMINI_FREE_KEY")
        if not API_KEY:
            raise Exception("Define la variable de entorno GEMINI_FREE_KEY con tu API key.")
        
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        if top_p > 1.0:
            top_p = 1.0
        response = response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_k": int(top_k),
                "top_p": top_p,
            }
        )

        return response.text
    
    @staticmethod
    def change_item_color(user_input: str, color_list:List[Tuple[int,int,int]]) -> str:
        prompt = PromptService.change_color_prompt(user_input, color_list)
        return GeminiChatService.chat(prompt)
    
    @staticmethod
    def select_action(user_input: str, temperature:float = 0.2, top_k:float = 8.0, top_p: float = 0.4) -> str:
        prompt = PromptService.select_action_prompt(user_input)
        return GeminiChatService.chat(prompt, temperature, top_k, top_p)
    
    @staticmethod
    def get_second_color(user_input: str) -> str:
        prompt = PromptService.get_second_color_prompt(user_input)
        return GeminiChatService.chat(prompt)
    
    @staticmethod
    def get_item_color(user_input: str, color_list:List[Tuple[int,int,int]]) -> str:
        prompt = PromptService.get_item_color_prompt(user_input, color_list)
        return GeminiChatService.chat(prompt)
    
    @staticmethod
    def get_new_color(user_input: str, model: str = "deepseek-r1:8b", think = False) -> str:
        prompt = PromptService.get_new_color_prompt(user_input)
        return GeminiChatService.chat(prompt)