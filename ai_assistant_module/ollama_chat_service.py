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
        # pedimos streaming
        stream = chat(
            model=model,
            messages=messages,
            stream=True,
            think=think,
            options={
                "temperature": 0.2,      # más bajo = más determinista
                "top_k": 8,              # restringe a los k tokens más probables
                "top_p": 0.4,            # nucleus sampling
            }
        )

        response_text = []
        print("Think:")
        for chunk in stream:
            # si el modelo emite "thinking", podés interceptarlo acá
            if hasattr(chunk.message, "thinking") and chunk.message.thinking:
                print(chunk.message.thinking, end="", flush=True)

            if chunk.message and chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                response_text.append(chunk.message.content)

        print()  # salto de línea final
        return "".join(response_text)
    
    @staticmethod
    def change_item_color(user_input: str, color_list:List[Tuple[int,int,int]], model: str = "deepseek-r1:8b", think = False) -> str:
        prompt = PromptService.change_color_prompt(user_input, color_list)
        return OllamaChatService.chat(prompt, model, think)
    
    @staticmethod
    def select_action(user_input: str, model: str = "deepseek-r1:8b", think = False) -> str:
        prompt = PromptService.select_action_prompt(user_input)
        return OllamaChatService.chat(prompt, model, think)
    
    @staticmethod
    def get_item(user_input: str, model: str = "deepseek-r1:8b", think = False) -> str:
        prompt = PromptService.get_item_prompt(user_input)
        return OllamaChatService.chat(prompt, model, think)
    
    @staticmethod
    def get_item_color(user_input: str, color_list:List[Tuple[int,int,int]], model: str = "deepseek-r1:8b", think = False) -> str:
        prompt = PromptService.get_item_color_prompt(user_input, color_list)
        return OllamaChatService.chat(prompt, model, think)
    
    @staticmethod
    def get_new_color(user_input: str, model: str = "deepseek-r1:8b", think = False) -> str:
        prompt = PromptService.get_new_color_prompt(user_input)
        return OllamaChatService.chat(prompt, model, think)
