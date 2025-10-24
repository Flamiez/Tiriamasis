from pydantic import BaseModel
from typing import List, Any

class ParsedImage(BaseModel):
    image_path: str
    person_id: int
    camera_id: str
    frame_id: int

class InputData(BaseModel):
    image_RGB: Any
    mask: Any
    color_palette: Any

class InputSequence(BaseModel):
    seqeunce: List[InputData]
    person_id: int
    camera_id: str

