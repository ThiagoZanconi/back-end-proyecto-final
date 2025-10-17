from pydantic import BaseModel, Field
from typing import Annotated

class ColorToRemove(BaseModel):
    color: Annotated[list[float], Field(min_length=3, max_length=3)]