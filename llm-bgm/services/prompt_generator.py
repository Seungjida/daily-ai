import openai
import re
from app.utils import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def preprocess_text(text: str) -> str:
    cleaned_text = re.sub(r"\b[ㄱ-ㅎㅏ-ㅣ]+\b", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()

def generate_prompt(diary_content: str) -> str:
    diary_content_cleaned = preprocess_text(diary_content)
    
    if not diary_content_cleaned:
        return "happy guitar tune, playful melody, cheerful tempo"
    
    prompt = (
        f"The following is a kindergarten diary entry:\n"
        f"'{diary_content_cleaned}'\n"
        "Based on the tone of this entry, create a concise music prompt. Use instruments with suitable tone ranges: "
        "For sad or calm tones, use instruments like piano, harp, flute, or strings with lower tones. For cheerful or energetic tones, use brighter instruments like ukulele, xylophone, or light percussion. "
        "Return the prompt as a short, comma-separated description without mood labels, like 'soft piano with low tones, soothing strings, slow tempo.'"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.8
    )

    return response.choices[0].message.content.strip()