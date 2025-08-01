from pathlib import Path

# 서버 설정
PORT = 8000

# 경로 설정
CKPT_PATH = Path("./tissue_model.ckpt")
CAPTURE_DIR = Path("./static/captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# 모델 설정
MODEL_CONFIG = {
    "image_size": (1024, 1024),
    "normalization": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }
} 