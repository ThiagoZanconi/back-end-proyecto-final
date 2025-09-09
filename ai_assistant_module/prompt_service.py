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
    def get_item_prompt(user_input: str) -> str:
        return (
            "You are an assistant that extracts the main item from a user's input.\n"
            "Return only the name of the item, nothing else.\n\n"
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
    def get_item_color_prompt(item: str, color_list:List[Tuple[int,int,int]]) -> str:
        return (
            "Return the RGB color that is most similar to the provided item \n"
            "Return only the RGB color of the item, nothing else.\n\n"
            f"RGB Color list: {color_list}\n"
            f"Item: {item}"
        )
    
    @staticmethod
    def get_new_color_prompt(user_input: str) -> str:
        return (
            "You are an assistant that extracts the RGB code of the main color from a user's input.\n"
            "Return only the RGB code of the color, nothing else.\n\n"
            "Example: [25,240,35]\n"
            f"User input: {user_input}"
        )