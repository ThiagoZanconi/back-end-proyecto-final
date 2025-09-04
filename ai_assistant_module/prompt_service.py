from typing import List, Tuple

class PromptService:
    
    @staticmethod
    def change_color_prompt(user_input: str, color_list: List[Tuple[int,int,int]]) -> str:
        return (
            f"You will be given a list of RGB colors and a user input.\n"
            f"Your task is to return the color from the list that best matches the user input.\n"
            f"Return only the color RGB code, nothing else.\n\n"
            f"The list of colors is: {color_list}\n"
            f"User input: {user_input}"
        )