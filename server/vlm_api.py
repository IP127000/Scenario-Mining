from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Union
from transformers import  Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import base64
import logging
import tempfile
import os
import shutil
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen3‑VL‑32B API",
    description="API for Qwen3‑VL‑32B (text / image / video)",
    version="0.1.0",
)

model_path = "/mnt/qwen3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    logger.info(" Model and processor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

class VideoUrlRequest(TextRequest):
    video_url: str  



async def save_upload_file_to_temp(upload_file: UploadFile, suffix: Optional[str] = None) -> str:
    suffix = suffix or os.path.splitext(upload_file.filename)[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd) 
    try:
        content = await upload_file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file to {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Failed to save upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")


def build_video_uri(file_path: str) -> str:
    abs_path = os.path.abspath(file_path)
    return abs_path


async def download_file_to_temp(url: str, suffix: str = ".mp4") -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download video, status {resp.status}")
                content = await resp.read()
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info(f"Downloaded video {url} -> {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Video download error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download video")

def clean_temp_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed temporary file {path}")
    except Exception as e:
        logger.warning(f"Failed to delete temp file {path}: {e}")

def generate_response(
    messages,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(DEVICE)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # temperature=temperature,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return output_text
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_from_video_upload")
async def generate_from_video_upload(
    text: str = Form(...),
    video: UploadFile = File(...),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.7),
    max_pixels: int = Form(1920 * 1080), 
    fps: Optional[float] = Form(None),
):
    temp_path = None
    try:
        temp_path = await save_upload_file_to_temp(video)
        video_uri = build_video_uri(temp_path)
        video_content = {"type": "video", "video": video_uri, "max_pixels": max_pixels}
        if fps is not None:
            video_content["fps"] = 1

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": text},
                ],
            }
        ]
        result = generate_response(messages, max_new_tokens, temperature)
        print(result)
        return {"result": result}
    finally:
        clean_temp_file(temp_path)

# 使用如下命令启动：
#  CUDA_VISIBLE_DEVICES=0,1   uvicorn your_module_name:app --host 0.0.0.0 --port 3000
