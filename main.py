import logging
import base64
import io
from PIL import Image
import numpy as np
import uvicorn
import cv2
from fastapi.responses import StreamingResponse
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyngrok import ngrok

# --- 설정 ---
from config import PORT, CAPTURE_DIR

# --- controller import ---
from controller import controller

# --- 로거 및 앱 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 서버 생명주기 관리 ---
@app.on_event("startup")
def startup_event():
    public_url = ngrok.connect(PORT)
    logger.info(f"--- Ngrok 터널이 생성되었습니다 ---")
    logger.info(f"Public URL: {public_url}")
    logger.info("---------------------------------")
    app.state.public_url = public_url
    
@app.on_event("shutdown")
def shutdown_event():
    ngrok.disconnect(app.state.public_url)
    logger.info("--- Ngrok 터널이 종료되었습니다 ---")

# --- 이미지 변환 함수 ---
def get_image_transforms():
    import albumentations as A
    return A.Compose([
        A.Resize(height=1024, width=1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# --- API 엔드포인트 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class PredictRequest(BaseModel):
    filename: str

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # 파일명으로 서버에 저장된 이미지 경로 생성
        file_path = CAPTURE_DIR / request.filename
        if not file_path.exists():
            return JSONResponse(status_code=404, content={"error": "파일을 찾을 수 없습니다."})
        
        # handler를 사용하여 추론 수행
        results = controller.run_inference(str(file_path))
        
        return {
            "success": True,
            "captured_image": results["captured_image_path"],
            "anomaly_map": results["anomaly_map_path"],
            "binary_mask": results["binary_mask_path"],
            "segmentation": results["segmentation_path"],
            "heatmap": results["heatmap_path"],
            "result_image": results["result_image_path"],
            "anomaly_score": results["anomaly_score"],
            "label": results["label"],
            "inference_time": results["inference_time"]
        }
    except Exception as e:
        logger.error(f"추론 중 오류 발생: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"서버 처리 중 오류 발생: {e}"})

@app.post("/can_scan")
async def check_can_scan(file: UploadFile = File(..., alias="file")):
    try:
        # FormData에서 이미지 파일 읽기
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(img)

        # 스캔 가능 여부 확인
        is_scannable, debug_msg = controller.can_scan(frame)
        return {"scannable": is_scannable, "debug_message": debug_msg}

    except Exception as e:
        logger.error(f"스캔 가능 여부 확인 중 오류 발생: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"서버 처리 중 오류 발생: {e}"})

@app.post("/scan_only")
async def scan_only(file: UploadFile = File(..., alias="file")):
    try:
        # FormData에서 이미지 파일 읽기
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(img)
        
        logger.info(f"스캔 시작: 이미지 크기 = {frame.shape}")

        # 스캔 가능 여부 먼저 확인
        can_scan_result, debug_msg = controller.can_scan(frame)
        logger.info(f"스캔 가능 여부: {can_scan_result}")
        logger.info(f"디버그 메시지: {debug_msg}")
        
        if not can_scan_result:
            logger.warning("스캔 가능한 객체를 찾지 못했습니다.")
            return JSONResponse(status_code=404, content={
                "error": "스캔할 수 있는 객체를 찾지 못했습니다.",
                "debug_message": debug_msg
            })

        # 스캔 및 리사이즈 수행 (controller가 저장까지 담당)
        scanned_image, saved_filepath = controller.scan_and_resize(frame)
        
        if scanned_image is not None and saved_filepath:
            logger.info(f"스캔된 이미지 저장 완료: {saved_filepath}")
            
            # 성공 시, 이미지를 JPEG 형식으로 인코딩하여 스트리밍으로 반환
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(scanned_image, cv2.COLOR_RGB2BGR))
            
            return {
                "success": True,
                "filename": Path(saved_filepath).name, # 파일명만 추출
                "image_data": base64.b64encode(buffer.tobytes()).decode('utf-8'),
                "debug_message": f"스캔 성공 및 저장: {saved_filepath}"
            }
        else:
            # 실패 시, 404 에러 반환
            logger.warning("스캔 처리 실패")
            return JSONResponse(status_code=404, content={
                "error": "스캔할 수 있는 객체를 찾지 못했습니다.",
                "debug_message": "컨트롤러에서 스캔 실패"
            })

    except Exception as e:
        logger.error(f"스캔 중 오류 발생: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"서버 처리 중 오류 발생: {e}"})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT) 