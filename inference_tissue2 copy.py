import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import PredictDataset

def detect_anomaly(ckpt_path: str, image_path: str, output_path: str):
    """
    PatchCore 체크포인트와 이미지를 사용하여 이상 탐지를 수행하고 결과를 저장합니다.

    Args:
        ckpt_path (str): .ckpt 모델 체크포인트 파일 경로
        image_path (str): 테스트할 이미지 파일 경로
        output_path (str): 결과 히트맵 이미지 저장 경로
    """
    # --- 1. Engine 및 모델 설정 ---
    print("1. Engine과 모델을 설정합니다...")
    
    # Engine 생성
    engine = Engine()
    
    # PatchCore 모델 생성 (체크포인트에서 로드)
    model = Patchcore.load_from_checkpoint(ckpt_path)
    print(f"  - 모델: {ckpt_path}")
    print(f"  - 추론 장치: {model.device}")

    # --- 2. 예측 데이터셋 생성 ---
    print(f"\n2. 예측 데이터셋을 생성합니다: {image_path}")
    dataset = PredictDataset(image_path)
    
    # --- 3. 추론 수행 ---
    print("\n3. 이상 탐지 추론을 수행합니다...")
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=ckpt_path,
        return_predictions=True
    )
    
    if predictions is None or len(predictions) == 0:
        raise RuntimeError("추론 결과가 없습니다.")
    
    # 첫 번째 예측 결과 사용
    prediction = predictions[0]
    print(f"  - 예측 결과 타입: {type(prediction)}")
    print(f"  - 예측 결과 키: {prediction.keys() if hasattr(prediction, 'keys') else 'No keys'}")

    # --- 4. 결과 분석 및 출력 ---
    print("\n4. 추론 결과를 분석하고 출력합니다.")
    
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
    
    # 임계값 설정 (기본값 0.5)
    threshold = 0.5
    label = "Anomalous" if score > threshold else "Normal"

    print(f"  - 이상 점수 (Anomaly Score): {score:.4f}")
    print(f"  - 판별 임계값 (Threshold): {threshold:.4f}")
    print(f"  - 최종 결과 (Label): {label}")
    print(f"  - 이상 맵 형태: {anomaly_map.shape}")

    # --- 5. 결과 시각화 및 저장 ---
    print(f"\n5. 결과 이미지를 생성하고 저장합니다: {output_path}")
    
    # 원본 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # 원본 이미지를 변환 크기에 맞게 리사이즈
    image_resized = cv2.resize(image, (1024, 1024))
    print(f"  - 원본 이미지 형태: {image_resized.shape}")
    
    # 이상치 맵을 히트맵으로 변환
    heatmap = (anomaly_map * 255).astype(np.uint8)
    print(f"  - 이상 맵 형태: {heatmap.shape}")
    
    # 히트맵을 3채널로 변환 (OpenCV는 BGR 사용)
    if len(heatmap.shape) == 2:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    else:
        # 이미 3채널인 경우 그대로 사용
        pass
    
    print(f"  - 히트맵 형태: {heatmap.shape}")
    
    # 히트맵을 원본 이미지와 같은 크기로 리사이즈
    if heatmap.shape[:2] != image_resized.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_resized.shape[1], image_resized.shape[0]))
        print(f"  - 리사이즈된 히트맵 형태: {heatmap.shape}")

    # 원본 이미지와 히트맵 합성
    overlay = cv2.addWeighted(image_resized, 0.6, heatmap, 0.4, 0)

    # 결과 이미지 저장
    cv2.imwrite(output_path, overlay)
    print("✓ 모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    # --- 커맨드 라인 인자 파서 설정 ---
    parser = argparse.ArgumentParser(
        description="PatchCore 모델로 이미지의 이상 여부를 탐지합니다."
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./results/patchcore_tissue/Patchcore/MVTecAD/tissue/v3/weights/lightning/model.ckpt", 
        help="모델 체크포인트(.ckpt) 파일의 경로"
    )
    parser.add_argument(
        "--image_path", type=str, default="scanned_20250730_144647_921ac756.jpg", 
        help="테스트할 JPG 이미지 파일의 경로"
    )
    parser.add_argument(
        "--output_path", type=str, default="inference_result.jpg", 
        help="결과 이미지를 저장할 경로"
    )
    args = parser.parse_args()

    detect_anomaly(args.ckpt_path, args.image_path, args.output_path)