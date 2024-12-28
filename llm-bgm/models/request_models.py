from pydantic import BaseModel

class PromptRequest(BaseModel):
    diary_content: str

class MusicRequest(BaseModel):
    bgm_prompt: str
    tokens: int = 750

class bgmRequest(BaseModel):
    diary_content: str
    tokens: int = 750
