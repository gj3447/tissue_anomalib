import cv2
import numpy as np
import os
from datetime import datetime

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    네 개의 점을 좌상단, 우상단, 우하단, 좌하단 순서로 정렬합니다.
    - 좌상단: x + y 합이 가장 작음
    - 우하단: x + y 합이 가장 큼
    - 우상단: y - x 차이가 가장 작음
    - 좌하단: y - x 차이가 가장 큼
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def scan_and_correct(image: np.ndarray) -> np.ndarray | None:
    """
    이미지에서 흰색 객체를 찾아 정면으로 보정하고 1024x1024 크기로 변환합니다.

    Args:
        image (np.ndarray): 스캔할 원본 이미지 (BGR NumPy 배열).

    Returns:
        np.ndarray | None: 변환된 1024x1024 이미지. 객체를 찾지 못하면 None을 반환합니다.
    """
    original_image = image.copy()

    # 1. 이미지 전처리 (검은 배경, 흰 객체에 최적화)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu의 이진화 방법을 사용하여 자동으로 임계값을 결정.
    # 흰색 객체와 검은 배경의 대비가 뚜렷할 때 효과적입니다.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 윤곽선 찾기
    # 가장 바깥쪽 윤곽선만 찾아서 처리 효율을 높입니다.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None

    # 면적이 가장 큰 윤곽선을 객체로 가정합니다.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    screen_contour = None

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        # 윤곽선을 근사화하여 꼭짓점 수를 줄입니다.
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        # 근사화된 윤곽선이 4개의 점을 가지면, 사각형으로 간주합니다.
        if len(approx) == 4:
            screen_contour = approx
            break
    
    if screen_contour is None:
        return None, None

    # 3. 투시 변환
    # 찾은 사각형의 꼭짓점 4개를 정렬합니다.
    rect = order_points(screen_contour.reshape(4, 2))
    
    # 변환될 결과 이미지의 목표 좌표 (1024x1024)
    dst = np.array([
        [0, 0],
        [1023, 0],
        [1023, 1023],
        [0, 1023]
    ], dtype="float32")
    
    # 원본의 4개 지점(rect)을 목표 지점(dst)으로 매핑하는 변환 행렬을 계산합니다.
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # 원본 이미지에 변환 행렬을 적용하여 최종 결과물을 생성합니다.
    warped = cv2.warpPerspective(original_image, M, (1024, 1024))
    
    return warped

# --- 함수 사용 예시 ---
if __name__ == '__main__':
    # 테스트할 이미지 파일 경로를 입력하세요.
    # 예: 'test_image.jpg'
    # 이 예제에서는 검은 배경에 흰색 A4 용지를 촬영한 이미지를 사용한다고 가정합니다.
    input_image_path = 'path/to/your/image.jpg'

    source_image = cv2.imread(input_image_path)

    if source_image is not None:
        print("이미지를 성공적으로 불러왔습니다. 스캔을 시작합니다...")
        scanned_image, saved_at = scan_and_correct(source_image)

        if scanned_image is not None:
            print(f"스캔 성공! 이미지가 다음 경로에 저장되었습니다: {saved_at}")
            
            # 원본과 결과 이미지를 화면에 표시
            cv2.imshow("Original Image", cv2.resize(source_image, (512, 512)))
            cv2.imshow("Scanned Image", scanned_image)
            
            print("아무 키나 누르면 창이 닫힙니다.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("이미지에서 사각형 객체를 찾는 데 실패했습니다.")
    else:
        print(f"이미지 파일을 찾을 수 없거나 읽을 수 없습니다: {input_image_path}")
        print("1. `input_image_path` 변수에 정확한 이미지 파일 경로를 입력했는지 확인하세요.")
        print("2. 'path/to/your/image.jpg'를 실제 파일 경로로 변경해야 합니다.")

