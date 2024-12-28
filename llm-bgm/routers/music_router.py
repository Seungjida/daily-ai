from fastapi import APIRouter, HTTPException
from app.models.request_models import MusicRequest
from app.services.music_generator import generate_music

router = APIRouter()

@router.post("/")
async def create_music(request: MusicRequest):
    try:
        s3_url = generate_music(request.bgm_prompt, request.tokens)
        return {"s3_url": s3_url, "message": "bgm이 생성 및 업로드 되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
