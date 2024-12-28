from fastapi import APIRouter, HTTPException
from app.models.request_models import bgmRequest
from app.models.response_models import bgmResponse
from app.services.bgm_generator import generate_bgm

router = APIRouter()

@router.post("/", response_model=bgmResponse)
async def create_bgm(request: bgmRequest):
    try:
        result = generate_bgm(request.diary_content, request.tokens)
        
        response = bgmResponse(
            bgm_prompt=result["bgm_prompt"],
            s3_url=result["s3_url"],
            message=result["message"]
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))