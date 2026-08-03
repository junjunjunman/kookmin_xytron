import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO

def extract_pizza_and_safe_path(model_path, input_folder, output_folder, csv_output_path):
    # 1. 모델 로드 및 폴더 확인
    if not os.path.exists(model_path):
        print(f"❌ 오류: 모델 파일({model_path})을 찾을 수 없습니다.")
        return
    
    print(f"🚀 모델 로드 중: {model_path}")
    model = YOLO(model_path)

    if not os.path.exists(input_folder):
        print(f"❌ 오류: 입력 폴더({input_folder})를 찾을 수 없습니다.")
        return

    os.makedirs(output_folder, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"⚠️ 경고: 입력 폴더에 이미지 파일이 없습니다.")
        return

    print(f"📂 총 {len(image_files)}개의 이미지를 처리합니다...")

    SLICE_STEP = (2 * np.pi) / 504
    csv_data = []

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            continue

        H, W = img.shape[:2]
        output_img = img.copy()
        
        # 2. 객체 탐지 수행
        results = model(img, conf=0.25, verbose=False)
        detections = results[0].boxes
        class_names = model.names

        current_points = []
        left_boxes = []
        right_boxes = []
        left_centroids = []
        right_centroids = []

        # 3. 객체 분류 및 바운딩 박스
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = str(class_names.get(cls_id, "")).lower()

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if 'current' in cls_name:
                current_points.append((x1, y1, x2, y2, conf))
                color = (0, 0, 255)
            elif 'left' in cls_name:
                left_boxes.append((x1, y1, x2, y2))
                left_centroids.append((cx, cy))
                color = (255, 255, 0)
            elif 'right' in cls_name:
                right_boxes.append((x1, y1, x2, y2))
                right_centroids.append((cx, cy))
                color = (255, 0, 0)

            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(output_img, (cx, cy), 5, color, -1)
            cv2.putText(output_img, f"{cls_name} {conf:.2f}", (x1, max(y1 - 5, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 4. current_point 중심점 (Cx, Cy) 설정
        if current_points:
            current_points.sort(key=lambda x: x[4], reverse=True)
            bx1, by1, bx2, by2, _ = current_points[0]
            Cx, Cy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
        else:
            Cx, Cy = W // 2, H
        cv2.drawMarker(output_img, (Cx, Cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # =========================================================
        # ⭐ [1] 중앙 안전 경로 생성 및 💡선분 각도 평균 계산 (업그레이드)
        # =========================================================
        left_centroids.sort(key=lambda x: x[1], reverse=True)
        right_centroids.sort(key=lambda x: x[1], reverse=True)

        central_path = []
        num_pairs = min(len(left_centroids), len(right_centroids))

        for i in range(num_pairs):
            L = left_centroids[i]
            R = right_centroids[i]
            cv2.line(output_img, L, R, (0, 165, 255), 1)
            
            mid_x = int((L[0] + R[0]) / 2)
            mid_y = int((L[1] + R[1]) / 2)
            central_path.append((mid_x, mid_y))
            cv2.circle(output_img, (mid_x, mid_y), 4, (0, 255, 255), -1)

        if len(central_path) > 1:
            for i in range(len(central_path) - 1):
                cv2.line(output_img, central_path[i], central_path[i+1], (0, 255, 255), 3)

        # 💡 제안하신 각도 평균화 로직 
        if len(central_path) > 0:
            # 시작점(로봇)을 포함한 전체 경로 리스트 생성
            path_points = [(Cx, Cy)] + central_path
            segment_angles = []
            
            # 각 선분의 각도(Degree) 계산
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                
                dx = p2[0] - p1[0]
                dy = p1[1] - p2[1] # 이미지 좌표계 특성상 위로 갈수록 y가 작아지므로 반대로 뺌
                dy = max(1, dy)    # 0으로 나누는 오류 방지
                
                angle_deg = np.degrees(np.arctan2(dx, dy))
                segment_angles.append(angle_deg)
            
            # 산출된 모든 선분 각도의 평균을 최종 조향각으로 사용 (Kp 불필요)
            steering_angle = float(np.mean(segment_angles))
            
        else:
            # 짝지어진 경로가 없을 때의 예외 처리
            if left_centroids:
                target_x = left_centroids[0][0] + 200 
            elif right_centroids:
                target_x = right_centroids[0][0] - 200
            else:
                target_x = Cx
            # 가상의 전방(150px) 목표점을 향한 각도 계산
            steering_angle = float(np.degrees(np.arctan2(target_x - Cx, 150)))

        # 조향각 제한 (-40 ~ 40)
        steering_angle = max(min(steering_angle, 40.0), -40.0)

        # =========================================================
        # ⭐ [2] 피자 조각 마스킹 (LiDAR 인덱스 범위)
        # =========================================================
        def get_angle(x, y):
            return np.arctan2(Cx - x, Cy - y)

        left_angles = []
        for (x1, y1, x2, y2) in left_boxes:
            left_angles.extend([get_angle(x, y) for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]])
        
        right_angles = []
        for (x1, y1, x2, y2) in right_boxes:
            right_angles.extend([get_angle(x, y) for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]])

        max_left_angle = max(left_angles) if left_angles else np.pi / 2
        min_right_angle = min(right_angles) if right_angles else -np.pi / 2

        max_left_slice_angle = np.ceil(max_left_angle / SLICE_STEP) * SLICE_STEP
        min_right_slice_angle = np.floor(min_right_angle / SLICE_STEP) * SLICE_STEP

        left_index = int(252 + (max_left_slice_angle / SLICE_STEP))
        right_index = int(252 + (min_right_slice_angle / SLICE_STEP))
        left_index = max(0, min(503, left_index))
        right_index = max(0, min(503, right_index))

        # 반투명 마스킹
        Y, X = np.ogrid[:H, :W]
        pixel_angles = np.arctan2(Cx - X, Cy - Y)
        outside_mask = ~((pixel_angles >= min_right_slice_angle) & (pixel_angles <= max_left_slice_angle))

        overlay = output_img.copy()
        overlay[outside_mask] = [30, 30, 30]
        blended = cv2.addWeighted(output_img, 0.6, overlay, 0.4, 0)
        output_img[outside_mask] = blended[outside_mask]

        line_len = max(H, W) * 2
        rx, ry = int(Cx - line_len * np.sin(min_right_slice_angle)), int(Cy - line_len * np.cos(min_right_slice_angle))
        lx, ly = int(Cx - line_len * np.sin(max_left_slice_angle)), int(Cy - line_len * np.cos(max_left_slice_angle))
        cv2.line(output_img, (Cx, Cy), (rx, ry), (0, 255, 255), 2)
        cv2.line(output_img, (Cx, Cy), (lx, ly), (0, 255, 255), 2)

        # =========================================================
        # ⭐ [3] 계산된 주행 경로 초록색 진한 선 표시
        # =========================================================
        draw_len = 250  
        end_x = int(Cx + draw_len * np.sin(np.radians(steering_angle)))
        end_y = int(Cy - draw_len * np.cos(np.radians(steering_angle)))
        
        cv2.line(output_img, (Cx, Cy), (end_x, end_y), (0, 255, 0), 4)
        cv2.circle(output_img, (end_x, end_y), 6, (0, 255, 0), -1)
        cv2.putText(output_img, f"Angle: {steering_angle:.1f} deg", (Cx + 20, Cy - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        save_path = os.path.join(output_folder, image_file)
        cv2.imwrite(save_path, output_img)
        print(f"✨ 완료: {image_file} (Angle: {steering_angle:.1f}°)")

        csv_data.append({
            "image_name": image_file,
            "right_index_start": right_index,
            "left_index_end": left_index,
            "target_angle": round(steering_angle, 2)
        })

    with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ["image_name", "right_index_start", "left_index_end", "target_angle"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\n✅ 완료! 결과 폴더: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    MODEL_FILE = "/Users/seongjun/Documents/국민대 자율주행/lidar.pt"
    INPUT_DIR = "/Users/seongjun/Documents/국민대 자율주행/lidar_cone"
    OUTPUT_DIR = "./pizza_slice_results"
    CSV_OUTPUT = "./lidar_index_ranges.csv"
    
    extract_pizza_and_safe_path(MODEL_FILE, INPUT_DIR, OUTPUT_DIR, CSV_OUTPUT)