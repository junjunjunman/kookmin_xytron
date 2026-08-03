import os
import cv2
import numpy as np
from ultralytics import YOLO

def generate_safe_path_for_folder(model_path, input_folder, output_folder):
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
        print(f"⚠️ 경고: 입력 폴더({input_folder})에 처리할 이미지 파일이 없습니다.")
        return

    print(f"📂 총 {len(image_files)}개의 이미지를 처리합니다...")

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"⚠️ 경고: 이미지를 읽을 수 없습니다: {image_file}")
            continue

        # 1. 객체 탐지 수행 (conf=0.25로 설정, 필요시 조절)
        results = model(img, conf=0.25, verbose=False)
        detections = results[0].boxes
        class_names = model.names

        left_points = []
        right_points = []

        # 원본 이미지 복사 (결과 시각화용)
        output_img = img.copy()

        # 2. 탐지된 모든 객체 확인 및 바운딩 박스/라벨 표시
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = class_names.get(cls_id) or class_names.get(str(cls_id))

            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            # 클래스별 분류 및 좌표 저장
            if cls_name and 'left' in str(cls_name).lower():
                left_points.append((centroid_x, centroid_y))
                box_color = (255, 255, 0) # Cyan
            elif cls_name and 'right' in str(cls_name).lower():
                right_points.append((centroid_x, centroid_y))
                box_color = (255, 0, 0) # Blue
            else:
                box_color = (0, 255, 0) # Green (current_point 등 기타)

            # 객체 인식 확인을 위한 바운딩 박스 및 텍스트 그리기
            cv2.rectangle(output_img, (x1, y1), (x2, y2), box_color, 2)
            label_text = f"{cls_name} {conf:.2f}"
            cv2.putText(output_img, label_text, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # 3. Y축 기준 정렬 (화면 아래가 로봇과 가까우므로 Y값이 클수록 아래쪽)
        left_points.sort(key=lambda x: x[1], reverse=True)
        right_points.sort(key=lambda x: x[1], reverse=True)

        # 4. left와 right끼리만 1대1 매칭하여 중앙점(Midpoint) 계산 후 선분 연결
        central_path = []
        num_pairs = min(len(left_points), len(right_points))

        for i in range(num_pairs):
            L = left_points[i]
            R = right_points[i]
            
            # Left와 Right 객체 중심을 직접 잇는 선 (선택사항: 주황색 점선 혹은 얇은 선)
            cv2.line(output_img, L, R, (0, 165, 255), 1)

            # Left와 Right 사이의 교점(중앙점) 계산
            mid_x = int((L[0] + R[0]) / 2)
            mid_y = int((L[1] + R[1]) / 2)
            central_path.append((mid_x, mid_y))

            # 중앙점에 작은 점 표시
            cv2.circle(output_img, (mid_x, mid_y), 4, (0, 255, 255), -1)

        # 5. 계산된 중앙점들을 순서대로 연결하여 '중앙 안전 경로' 선분 생성
        if len(central_path) > 1:
            for i in range(len(central_path) - 1):
                cv2.line(output_img, central_path[i], central_path[i+1], (0, 255, 255), 3)
            
            # 안내 텍스트
            text_pt = central_path[0]
            cv2.putText(output_img, 'Central Safe Path', (text_pt[0] - 60, text_pt[1] + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 결과 저장
        save_path = os.path.join(output_folder, image_file)
        cv2.imwrite(save_path, output_img)
        print(f"✨ 처리 및 저장 완료: {image_file} (Left: {len(left_points)}개, Right: {len(right_points)}개)")

    print(f"\n✅ 모든 작업 완료! 결과 폴더: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    MODEL_FILE = "/Users/seongjun/Documents/국민대 자율주행/lidar.pt"
    INPUT_DIR = "/Users/seongjun/Documents/국민대 자율주행/lidar_cone"
    OUTPUT_DIR = "./safe_path_results"
    
    generate_safe_path_for_folder(MODEL_FILE, INPUT_DIR, OUTPUT_DIR)