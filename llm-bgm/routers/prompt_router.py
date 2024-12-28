from fastapi import APIRouter, HTTPException
from app.models.request_models import PromptRequest
from app.services.prompt_generator import generate_prompt

router = APIRouter()

@router.post("/")
async def create_prompt(request: PromptRequest):
    try:
        bgm_prompt  = generate_prompt(request.diary_content)
        return {"bgm_prompt": bgm_prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
