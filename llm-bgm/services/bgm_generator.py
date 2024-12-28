from app.services.prompt_generator import generate_prompt
from app.services.music_generator import generate_music

def generate_bgm(diary_content: str, tokens: int) -> dict:
    bgm_prompt = generate_prompt(diary_content)
    s3_url = generate_music(bgm_prompt, tokens)
    
    return {
        "bgm_prompt": bgm_prompt,
        "s3_url": s3_url,
        "message": "BGM이 성공적으로 생성 및 업로드 되었습니다."
    }
