from pydantic import BaseModel

class ParsedImage(BaseModel):
    image_path: str
    person_id: int
    camera_id: str
    frame_id: int
