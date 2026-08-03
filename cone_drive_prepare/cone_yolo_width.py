import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO

def analyze_and_save_cone_segmentation(model_path, input_folder, output_img_folder, csv_output_path):
    # 1. 모델 검증 및 로드
    if not os.path.exists(model_path):
        print(f"❌ 오류: 모델 파일({model_path})을 찾을 수 없습니다.")
        return
    
    print(f"🚀 Segmentation 모델 로드 중: {model_path}")
    model = YOLO(model_path)

    # 2. 입력 및 출력 폴더 검증/생성
    if not os.path.exists(input_folder):
        print(f"❌ 오류: 입력 폴더({input_folder})를 찾을 수 없습니다.")
        return

    os.makedirs(output_img_folder, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"⚠️ 경고: 입력 폴더({input_folder})에 이미지 파일이 없습니다.")
        return

    print(f"📂 총 {len(image_files)}개 이미지 추론 및 분석 시작...")

    results_data = []

    # 3. 이미지별 추론 및 개별 넓이 계산 Loop
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        # 모델 추론 (save=True 옵션 대신 직접 그리기 위해 결과 객체 받음)
        results = model.predict(source=image_path, conf=0.25, verbose=False)
        result = results[0]

        # 🌟 YOLO 적용 결과 이미지 저장 (세그멘테이션 마스크 및 바운딩박스가 그려진 이미지)
        annotated_img = result.plot()
        save_img_path = os.path.join(output_img_folder, image_file)
        cv2.imwrite(save_img_path, annotated_img)

        cone_details = []
        total_area = 0

        # 마스크(Segmentation Mask) 정보 확인
        if result.masks is not None:
            boxes = result.boxes
            masks = result.masks.data.cpu().numpy()  # Binary masks (N, H, W)
            class_names = model.names

            cone_idx = 1
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = class_names.get(cls_id) or class_names.get(str(cls_id))

                # 'cone' 객체만 필터링
                if cls_name and 'cone' in str(cls_name).lower():
                    # 마스크 픽셀 수 계산
                    mask = masks[idx]
                    area = int(np.sum(mask > 0.5))
                    
                    # 개별 객체 ID와 면적 저장
                    cone_details.append({
                        "cone_id": f"cone_{cone_idx}",
                        "area_pixels": area
                    })
                    total_area += area
                    cone_idx += 1

        # 개별 면적들을 보기 쉬운 문자열 형태로 정렬 ("cone_1: 1200 | cone_2: 850")
        details_str = " | ".join([f"{item['cone_id']}: {item['area_pixels']}px" for item in cone_details])

        results_data.append({
            "image_name": image_file,
            "cone_count": len(cone_details),
            "total_area_pixels": total_area,
            "individual_areas": details_str,
            # 오름차순 정렬용 (첫 번째 콘의 넓이, 없으면 0)
            "first_cone_area": cone_details[0]["area_pixels"] if cone_details else 0 
        })

    # 4. 넓이 기준 오름차순 정렬 (총 넓이 기준)
    results_data.sort(key=lambda x: x["total_area_pixels"])

    # 5. CSV 파일로 저장
    with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ["image_name", "cone_count", "total_area_pixels", "individual_areas"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in results_data:
            # 정렬 임시 키 제거 후 쓰기
            del row["first_cone_area"]
            writer.writerow(row)

    print(f"\n✅ 작업 완료!")
    print(f"🖼️ YOLO 결과 이미지 저장 폴더: {os.path.abspath(output_img_folder)}")
    print(f"📊 CSV 파일 저장 위치: {os.path.abspath(csv_output_path)}")

if __name__ == "__main__":
    # --- 경로 설정 ---
    MODEL_FILE = "/Users/seongjun/Documents/국민대 자율주행/best_records_0724.pt"
    INPUT_DIR = "/Users/seongjun/Documents/국민대 자율주행/cone"
    
    OUTPUT_IMG_DIR = "./yolo_segmented_images"  # YOLO 적용 사진이 저장될 폴더
    CSV_OUTPUT = "./cone_individual_areas.csv"   # CSV 저장 경로

    # 실행
    analyze_and_save_cone_segmentation(MODEL_FILE, INPUT_DIR, OUTPUT_IMG_DIR, CSV_OUTPUT)