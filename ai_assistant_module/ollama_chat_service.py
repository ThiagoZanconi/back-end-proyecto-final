from typing import List, Tuple
from ollama import chat

from ai_assistant_module.prompt_service import PromptService

class OllamaChatService:

    @staticmethod
    def chat(prompt: str, model: str = "deepseek-r1:8b", think = False) -> str:
        messages = [
            {
                "role": "user", 
                "content": prompt
            },
        ]

        response = chat(model, messages, think=think)

        return response.message.content
    
    @staticmethod
    def change_color(user_input: str, color_list:List[Tuple[int,int,int]], model: str = "deepseek-r1:8b", think = False) -> str:
        prompt = PromptService.change_color_prompt(user_input, color_list)
        return OllamaChatService.chat(prompt, model, think)