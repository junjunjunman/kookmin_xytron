import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO

def process_images():
    input_dir = "img"
    output_dir = "img_yolo"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 캘리브레이션 파라미터 및 모델 로드
    data = np.load("camera_params.npz")
    camera_matrix, dist_coeffs = data["mtx"], data["dist"]
    model = YOLO("best.pt")

    # 2. 클래스별 색상 매핑 (OpenCV는 BGR 기준)
    color_map = {
        'left': (235, 206, 135),   # 하늘색 (Sky Blue)
        'right': (255, 255, 0),    # 청록색 (Cyan)
        'yellow': (0, 255, 255),   # 노란색 (Yellow)
        'cone': (0, 165, 255)      # 주황색 (Orange)
    }

    # 이미지 탐색
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: 
            continue
        
        # 3. 왜곡 보정
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        rx, ry, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[ry:ry+rh, rx:rx+rw]

        # 4. YOLO 세그멘테이션 예측
        results = model.predict(undistorted, conf=0.4, verbose=False)[0]
        
        # 5. 배경 및 오버레이 초기화
        overlay = undistorted.copy()
        black_bg = np.zeros_like(undistorted)

        if results.masks is not None:
            for i, poly in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i].item())
                label = model.names[cls_id]
                
                # 지정되지 않은 클래스는 기본값(흰색) 적용
                color = color_map.get(label, (255, 255, 255))
                
                pts = poly.astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.fillPoly(black_bg, [pts], color)
                
        # 왼쪽 이미지는 반투명하게 원본과 섞기
        blended_left = cv2.addWeighted(overlay, 0.5, undistorted, 0.5, 0)
        
        # 6. 좌(오버레이) - 우(검은배경) 이어붙이기
        combined_img = np.hstack((blended_left, black_bg))
        
        # 7. 저장
        fname = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, fname), combined_img)
        print(f"[{fname}] 변환 및 병합 저장 완료")

    print(f"\n작업 완료! 결과는 '{output_dir}/' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    process_images()