from pydantic import BaseModel, Field
from typing import Annotated

class GammaRequest(BaseModel):
    color: Annotated[list[int], Field(min_length=3, max_length=3)]
    new_color: Annotated[list[int], Field(min_length=3, max_length=3)]