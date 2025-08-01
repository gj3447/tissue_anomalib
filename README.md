# 이상 탐지 시스템 (Anomaly Detection System)

본 프로젝트는 FastAPI를 기반으로 구축된 이상 탐지 시스템입니다. 이미지에서 특정 객체를 자동으로 스캔 및 보정하고, 딥러닝 모델을 사용하여 이상 여부를 판별합니다.

## 주요 기능

- **문서/객체 스캔**: 이미지 내에서 객체를 자동으로 감지하고, 정면에서 바라보는 시점으로 투시 변환합니다.
- **이상 탐지**: 딥러닝 모델을 활용하여 스캔된 이미지의 정상/비정상 여부를 판별합니다.
- **웹 인터페이스**: 웹을 통해 실시간으로 이미지를 스트리밍하고, 스캔 및 추론 결과를 확인할 수 있습니다.

## 기술 스택

- **백엔드**: FastAPI
- **머신러닝**: PyTorch (모델 파일 기반)
- **이미지 처리**: OpenCV
- **서버**: Uvicorn

## 프로젝트 구조

```
/
├── .venv/                   # 가상 환경
├── scanned_images/          # 스캔 후 보정된 이미지가 저장되는 폴더
├── static/                  # CSS, JS 등 정적 파일
├── templates/               # HTML 템플릿 파일
│   └── index.html           # 메인 페이지
├── .gitattributes           # Git LFS 추적 파일 설정
├── config.py                # 설정 변수 (모델 경로 등)
├── controller.py            # 비즈니스 로직 처리 (스캔, 추론)
├── document_scanner.py      # 문서/객체 스캔 및 보정 기능
├── main.py                  # FastAPI 앱 실행 및 라우팅
├── model_handler.py         # 딥러닝 모델 로드 및 추론 처리
├── requirements.txt         # 프로젝트 의존성 목록
├── tissue_model.ckpt        # (LFS) 이상 탐지 모델 파일
└── README.md                # 프로젝트 설명서
```

## API 엔드포인트

- `GET /`: 메인 웹 페이지를 렌더링합니다.
- `GET /video_feed`: 실시간 카메라 영상을 스트리밍하는 웹소켓 엔드포인트입니다.
- `POST /capture`: 현재 프레임을 캡처하고 객체 스캔을 수행합니다. 스캔 성공 시, 보정된 이미지와 파일 경로를 반환합니다.
- `POST /inference`: 지정된 이미지 파일에 대해 이상 탐지 추론을 실행하고 결과를 반환합니다.

## 동작 흐름 (Workflow)

1.  **웹 페이지 접속**: 사용자가 메인 페이지(`GET /`)에 접속합니다.
2.  **실시간 스트리밍**: 서버는 웹소켓(`/video_feed`)을 통해 카메라 영상을 실시간으로 클라이언트에 전송합니다.
3.  **객체 스캔**: 사용자가 '캡처' 버튼을 누르면, 현재 영상 프레임이 서버의 `/capture` 엔드포인트로 전송됩니다.
    - `controller.py`는 `document_scanner.py`를 호출하여 이미지 내에서 객체를 찾고 정면으로 보정합니다.
    - 보정된 이미지는 `scanned_images/` 폴더에 저장됩니다.
4.  **이상 탐지 추론**: 스캔이 성공하면, 클라이언트는 저장된 이미지 경로를 사용하여 `/inference` 엔드포인트로 추론을 요청합니다.
    - `controller.py`는 `model_handler.py`를 통해 `tissue_model.ckpt` 모델을 사용하여 이상 점수를 계산합니다.
5.  **결과 확인**: 추론 결과(이상 점수, 히트맵 이미지 등)가 클라이언트에 반환되어 화면에 표시됩니다.

## 설치 및 실행 방법

### 1. 저장소 복제

```bash
git clone <your-repository-url>
cd AnomalyDetection3/app
```

### 2. Git LFS 설치

본 프로젝트는 대용량 모델 파일(`.ckpt`)을 관리하기 위해 Git LFS를 사용합니다. 반드시 Git LFS를 설치한 후 저장소를 `clone`하거나, 이미 `clone`했다면 아래 명령어를 실행하여 LFS 파일을 받아오세요.

```bash
# Git LFS 설치 (최초 1회)
git lfs install

# LFS 파일 다운로드
git lfs pull
```

### 3. 가상 환경 및 의존성 설치

프로젝트 실행에 필요한 라이브러리들을 설치합니다.

```bash
# Conda 가상환경 생성 (예시)
conda create -n anom-env python=3.9
conda activate anom-env

# 필요 라이브러리 설치
pip install -r requirements.txt
```

### 4. 서버 실행

Uvicorn을 사용하여 FastAPI 애플리케이션을 실행합니다. `--reload` 옵션을 사용하면 코드 변경 시 서버가 자동으로 재시작됩니다.

```bash
uvicorn main:app --reload
```

서버가 실행되면 `http://127.0.0.1:8000` 주소로 접속하여 확인할 수 있습니다.
