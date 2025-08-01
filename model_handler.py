import logging
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from config import CKPT_PATH, CAPTURE_DIR

logger = logging.getLogger(__name__)

class AnomalyModelHandler:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.engine = Engine()
        self.model = self._load_model(ckpt_path)

    def _load_model(self, ckpt_path):
        logger.info(f"모델을 불러옵니다: {ckpt_path}")
        if not Path(ckpt_path).exists():
            logger.error(f"모델 파일이 존재하지 않습니다: {ckpt_path}")
            return None
        
        model = Patchcore.load_from_checkpoint(ckpt_path)
        logger.info(f"사용할 추론 장치: {model.device}")
        logger.info("모델 불러오기 완료.")
        return model

    def run_inference(self, image_path: str):
        """이미지 경로를 입력으로 받아 추론을 수행합니다."""
        if not self.model:
            raise RuntimeError("모델이 준비되지 않았습니다.")

        logger.info(f"추론할 이미지 경로: {image_path}")
        
        # 이미지 파일 존재 확인
        if not Path(image_path).exists():
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")

        # 2. Engine을 사용한 추론
        logger.info("--- Engine 추론 시작 ---")
        try:
            # PredictDataset 생성
            dataset = PredictDataset(image_path)
            logger.info(f"데이터셋 생성 완료: {len(dataset)} 개의 이미지")
            
            # Engine을 사용한 추론
            start_time = time.time()
            predictions = self.engine.predict(
                model=self.model,
                dataset=dataset,
                ckpt_path=self.ckpt_path,
                return_predictions=True
            )
            end_time = time.time()
            inference_time = end_time - start_time
            logger.info(f"--- Engine 추론 완료 (소요 시간: {inference_time:.4f}초) ---")
            
            if predictions is None or len(predictions) == 0:
                raise RuntimeError("추론 결과가 없습니다.")
            
            # 첫 번째 예측 결과 사용
            prediction = predictions[0]
            logger.info(f"추론 결과 타입: {type(prediction)}")
            logger.info(f"추론 결과 키: {prediction.keys() if hasattr(prediction, 'keys') else 'No keys'}")
            
        except Exception as e:
            logger.error(f"추론 중 오류 발생: {e}", exc_info=True)
            raise

        # 3. 결과 분석 및 시각화
        logger.info("--- 결과 처리 시작 ---")
        
        try:
            # 결과에서 점수와 이상 맵 추출
            if hasattr(prediction, 'pred_score'):
                score = prediction.pred_score.item()
            elif 'pred_score' in prediction:
                score = prediction['pred_score'].item()
            else:
                score = 0.5  # 기본값
            
            if hasattr(prediction, 'anomaly_map'):
                anomaly_map = prediction.anomaly_map.squeeze().cpu().numpy()
            elif 'anomaly_map' in prediction:
                anomaly_map = prediction['anomaly_map'].squeeze().cpu().numpy()
            else:
                # 스칼라 점수를 2D 맵으로 변환
                anomaly_map = np.full((1024, 1024), score)
            
            logger.info(f"이상 점수: {score:.4f}")
            logger.info(f"이상 맵 형태: {anomaly_map.shape}")
            logger.info(f"이상 맵 값 범위: min={anomaly_map.min()}, max={anomaly_map.max()}")

            # 임계값 설정 (기본값 0.5)
            threshold = 0.5
            pred_label_bool = score < threshold

            # 원본 이미지 로드 (1024x1024로 리사이즈)
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            # BGR -> RGB 변환
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_resized = cv2.resize(original_image_rgb, (1024, 1024))

            # 3-1. 이상치 맵을 1024x1024로 리사이즈
            anomaly_map_resized = cv2.resize(anomaly_map, (1024, 1024))
            logger.info(f"리사이즈된 이상치 맵: shape={anomaly_map_resized.shape}")
            
            # 결과 이미지들을 저장할 폴더 생성
            image_stem = Path(image_path).stem
            result_dir = CAPTURE_DIR / "results" / image_stem
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 3-2. 이상치 맵 (그레이스케일) 저장
            anomaly_map_normalized = (anomaly_map_resized * 255).astype(np.uint8)
            anomaly_map_path = result_dir / "anomaly_map.png"
            cv2.imwrite(str(anomaly_map_path), anomaly_map_normalized)
            logger.info(f"이상치 맵 저장 완료: {anomaly_map_path}")
            
            # 3-3. 이진 마스크 생성 (임계값 사용)
            binary_mask = (anomaly_map_resized > threshold).astype(np.uint8) * 255
            binary_mask_path = result_dir / "binary_mask.png"
            cv2.imwrite(str(binary_mask_path), binary_mask)
            logger.info(f"이진 마스크 저장 완료: {binary_mask_path}")
            
            # 3-4. 세그멘테이션 결과 (원본에 마스크 오버레이)
            # 마스크를 컬러로 변환 (빨간색으로 불량 부분 표시)
            mask_colored = np.zeros_like(original_image_resized)
            mask_colored[binary_mask > 0] = [255, 0, 0]  # 빨간색으로 불량 부분 표시
            
            # 원본과 마스크 합성
            segmentation_result = cv2.addWeighted(original_image_resized, 0.7, mask_colored, 0.3, 0)
            segmentation_path = result_dir / "segmentation.png"
            cv2.imwrite(str(segmentation_path), cv2.cvtColor(segmentation_result, cv2.COLOR_RGB2BGR))
            logger.info(f"세그멘테이션 결과 저장 완료: {segmentation_path}")
            
            # 3-5. 히트맵 생성
            heatmap = cv2.applyColorMap((anomaly_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_path = result_dir / "heatmap.png"
            cv2.imwrite(str(heatmap_path), heatmap)
            logger.info(f"히트맵 저장 완료: {heatmap_path}")
            
            # 3-6. 히트맵과 원본 합성
            result_image_cv = cv2.addWeighted(original_image_resized, 0.6, heatmap, 0.4, 0)
            result_image_path = result_dir / "result.png"
            cv2.imwrite(str(result_image_path), cv2.cvtColor(result_image_cv, cv2.COLOR_RGB2BGR))
            logger.info(f"합성 결과 저장 완료: {result_image_path}")

            logger.info("--- 모든 처리 완료 ---")
            
            # 반환 경로를 웹에서 접근 가능한 static 경로로 변환
            # (e.g., C:\...\static\captures -> static/captures)
            base_web_path = Path("static") / "captures" / "results" / image_stem

            return {
                "captured_image_path": image_path,
                "anomaly_map_path": str(base_web_path / "anomaly_map.png").replace("\\", "/"),
                "binary_mask_path": str(base_web_path / "binary_mask.png").replace("\\", "/"),
                "segmentation_path": str(base_web_path / "segmentation.png").replace("\\", "/"),
                "heatmap_path": str(base_web_path / "heatmap.png").replace("\\", "/"),
                "result_image_path": str(base_web_path / "result.png").replace("\\", "/"),
                "anomaly_score": float(score),
                "label": "Normal" if pred_label_bool else "Anomalous",
                "inference_time": f"{inference_time:.4f}"
            }
            
        except Exception as e:
            logger.error(f"결과 처리 중 오류 발생: {e}", exc_info=True)
            raise 