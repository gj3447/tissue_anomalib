import logging
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from document_scanner import scan_and_correct
from model_handler import AnomalyModelHandler
from config import CKPT_PATH, CAPTURE_DIR

logger = logging.getLogger(__name__)

class AnomalyController:
    def __init__(self):
        """컨트롤러 초기화"""
        self.model_handler = AnomalyModelHandler(CKPT_PATH)
        logger.info("AnomalyController 초기화 완료")

    def can_scan(self, frame: np.ndarray) -> tuple[bool, str]:
        """이미지에서 스캔 가능한 사각형 객체를 찾을 수 있는지 확인하고, 결과와 메시지를 튜플로 반환합니다."""
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            scanned_image = scan_and_correct(frame_bgr)
            
            if scanned_image is not None:
                return True, "스캔 가능한 객체를 찾았습니다."
            else:
                return False, "스캔 가능한 객체를 찾지 못했습니다."
        except Exception as e:
            logger.error(f"스캔 가능 여부 확인 중 오류: {e}")
            return False, f"오류 발생: {e}"

    def scan_and_resize(self, frame: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        """이미지에서 사각형을 스캔, 변환 후 저장하고 결과를 반환합니다."""
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            scanned_bgr = scan_and_correct(frame_bgr)

            if scanned_bgr is not None:
                # 파일 저장 로직
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"scanned_{timestamp}.jpg"
                filepath = os.path.join(CAPTURE_DIR, filename)
                
                # 디렉토리가 없으면 생성
                os.makedirs(CAPTURE_DIR, exist_ok=True)
                
                # OpenCV는 BGR 형식으로 이미지를 저장합니다.
                cv2.imwrite(filepath, scanned_bgr)

                # UI 표시를 위해 RGB로 변환하여 반환
                scanned_rgb = cv2.cvtColor(scanned_bgr, cv2.COLOR_BGR2RGB)
                return scanned_rgb, filepath
            else:
                return None, None
        except Exception as e:
            logger.error(f"스캔 및 리사이즈 중 오류: {e}")
            return None, None

    def run_inference(self, image_path: str):
        """이미지 경로를 입력으로 받아 추론을 수행합니다."""
        return self.model_handler.run_inference(image_path)

# --- 전역 컨트롤러 인스턴스 ---
# 서버 시작 시 한 번만 생성됨
controller = AnomalyController() 