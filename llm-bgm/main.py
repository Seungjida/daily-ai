from fastapi import FastAPI
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from app.routers import prompt_router, music_router, bgm_router

app = FastAPI()

app.include_router(prompt_router, prefix="/generate-prompt")
app.include_router(music_router, prefix="/generate-music")
app.include_router(bgm_router, prefix="/generate-bgm")
