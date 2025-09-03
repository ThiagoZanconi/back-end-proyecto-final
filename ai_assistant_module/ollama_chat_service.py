from ollama import chat

class OllamaChatService:

    @staticmethod
    def chat(prompt: str, model: str = "deepseek-r1:8b") -> str:
        messages = [
            {
                "role": "user", 
                "content": prompt
            },
        ]

        response = chat(model, messages, think=False)

        return response.message.content