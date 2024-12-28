from pydantic import BaseModel

class bgmResponse(BaseModel):
    bgm_prompt: str
    s3_url: str
    message: str
