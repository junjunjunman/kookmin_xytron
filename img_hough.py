import cv2
import numpy as np
import os
import glob
import math
from ultralytics import YOLO

def process_images():
    input_dir = "img"
    output_dir = "img_hough"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 캘리브레이션 파라미터 및 모델 로드
    data = np.load("camera_params.npz")
    camera_matrix, dist_coeffs = data["mtx"], data["dist"]
    model = YOLO("best.pt")

    # 2. Top-view 변환용 파라미터 설정
    before = np.array([[77, 389], [616, 402], [506, 330], [201, 324]], dtype='float32')
    after = np.array([[100, 300], [400, 300], [400, 100], [100, 100]], dtype='float32')
    M = cv2.getPerspectiveTransform(before, after)
    tdSize = (500, 350)

    color_map = {
        'left': (235, 206, 135),
        'right': (255, 255, 0),
        'yellow': (0, 255, 255),
        'cone': (0, 165, 255)
    }
    target_labels = ['left', 'right', 'yellow']

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
            h, w = undistorted.shape[:2]

        # 4. YOLO 세그멘테이션
        results = model.predict(undistorted, conf=0.4, verbose=False)[0]
        
        overlay = undistorted.copy()
        black_bg = np.zeros_like(undistorted)
        class_masks = {lbl: np.zeros((h, w), dtype=np.uint8) for lbl in target_labels}

        if results.masks is not None:
            for i, poly in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i].item())
                label = model.names[cls_id]
                color = color_map.get(label, (255, 255, 255))
                
                pts = poly.astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.fillPoly(black_bg, [pts], color)
                
                if label in target_labels:
                    cv2.fillPoly(class_masks[label], [pts], 255)
                
        # 왼쪽: 왜곡 보정 원본 + 오버레이
        left_img = cv2.addWeighted(overlay, 0.5, undistorted, 0.5, 0)
        
        # 오른쪽: 검은 배경 오버레이를 Top-view로 변환
        right_img = cv2.warpPerspective(black_bg, M, tdSize)
        
        # 5. 타겟 클래스별 Hough 변환 및 각도 계산
        angles = []
        for label in target_labels:
            mask_td = cv2.warpPerspective(class_masks[label], M, tdSize)
            edges = cv2.Canny(mask_td, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=20)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line.flatten()
                    
                    # y좌표가 큰 쪽(아래쪽)을 (x1, y1)으로 설정
                    if y1 < y2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                        
                    dx = x2 - x1
                    dy = y1 - y2  # 위로 향할수록 양수
                    
                    if dy > 0:
                        angle = math.degrees(math.atan2(dx, dy))
                        angles.append(angle)
                        # 빨간색으로 추출된 직선 그리기
                        cv2.line(right_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 6. 최종 방향 도출 및 시각화
        final_angle = 0.0
        if angles:
            final_angle = np.mean(angles)
            # 조건에 따라 -40 ~ 40 도로 제한
            final_angle = np.clip(final_angle, -40, 40)
            
        # 최종 라인 그리기 (초록색)
        start_x, start_y = tdSize[0] // 2, tdSize[1]
        length = 150
        end_x = int(start_x + length * math.sin(math.radians(final_angle)))
        end_y = int(start_y - length * math.cos(math.radians(final_angle)))
        cv2.line(right_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
        
        # 텍스트 출력
        cv2.putText(right_img, f"Angle: {final_angle:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 7. 해상도 맞춤 및 좌우 병합
        rh, rw = right_img.shape[:2]
        new_rw = int(rw * (h / rh))
        right_img_resized = cv2.resize(right_img, (new_rw, h))
        
        combined_img = np.hstack((left_img, right_img_resized))
        
        fname = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, fname), combined_img)
        print(f"[{fname}] 처리 완료 (Angle: {final_angle:.1f}°)")

    print(f"\n작업 완료! 결과는 '{output_dir}/' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    process_images()