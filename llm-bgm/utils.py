import boto3
from dotenv import load_dotenv
import os

load_dotenv()

s3_client = boto3.client("s3")
bucket_name = os.getenv("S3_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not bucket_name:
    raise ValueError("S3_BUCKET_NAME 환경 변수가 세팅되지 않았습니다.")

def upload_to_s3(file_name: str, object_name: str) -> str:
    print(f"Uploading file: {file_name} to bucket: {bucket_name} as {object_name}")

    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"{file_name} 파일이 존재하지 않습니다.")
    
    try: 
        s3_client.upload_file(file_name, bucket_name, object_name)
        return f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
    except Exception as e:
        print(f"{file_name}를 S3에 올리는데 실패하였습니다.: {e}")
        raise

