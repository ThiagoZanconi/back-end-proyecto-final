from typing import List, Tuple

class PromptService:

    ACTION_LIST = [
        "1: Change item color",
        "2: Change background color",
    ]
    
    @staticmethod
    def change_color_prompt(user_input: str, color_list: List[Tuple[int,int,int]]) -> str:
        return (
            f"You will be given a list of RGB colors and a user input.\n"
            f"Your task is to return the color from the list that best matches the user input.\n"
            f"Return only the color RGB code, nothing else.\n\n"
            f"The list of colors is: {color_list}\n"
            f"User input: {user_input}"
        )
    
    @staticmethod
    def select_action_prompt(user_input: str) -> str:
        return (
            "You are an assistant that helps users choose between a list of actions, from a user's input.\n"
            "Based on the user's input, determine which action is more appropriate.\n"
            "Return only the number corresponding to the selected action, nothing else.\n\n"
            f"User input: {user_input}\n"
            f"Action list: {PromptService.ACTION_LIST}"
        )
    
    @staticmethod
    def get_item_prompt(user_input: str, color_list:List[Tuple[int,int,int]]) -> str:
        return (
            "You are an assistant that extracts the color of the main item from a user's input.\n"
            "Return only the rgb color of the item, nothing else.\n\n"
            f"User input: {user_input}"
            f"The list of colors is: {color_list}\n"
        )
    
    @staticmethod
    def get_color_prompt(user_input: str) -> str:
        return (
            "You are an assistant that extracts the main color from a user's input.\n"
            "Return only the rgb code of the color, nothing else.\n\n"
            "Example: [25,240,35]\n"
            f"User input: {user_input}"
        )