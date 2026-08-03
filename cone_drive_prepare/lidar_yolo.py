import os
from ultralytics import YOLO

def run_yolo_inference(model_path, input_folder, output_folder):
    """
    지정한 폴더의 이미지들에 YOLOv11 모델을 적용하고 결과를 저장합니다.
    """
    # 1. 모델 로드 (lidar.pt 경로)
    if not os.path.exists(model_path):
        print(f"❌ 오류: 모델 파일({model_path})을 찾을 수 없습니다.")
        return
    
    print(f"🚀 모델 로드 중: {model_path}")
    model = YOLO(model_path)

    # 2. 입력 폴더 확인
    if not os.path.exists(input_folder):
        print(f"❌ 오류: 입력 폴더({input_folder})를 찾을 수 없습니다.")
        return

    print(f"📂 입력 폴더 감지: {input_folder}")
    print(f"🔄 처리 중... 잠시만 기다려주세요.")

    # 3. 모델 예측(Predict) 실행 및 결과 저장
    # project와 name 속성을 이용해 결과가 저장될 폴더 지정
    results = model.predict(
        source=input_folder,
        project=output_folder,
        name="result_images",
        save=True,      # 결과 이미지 저장 활성화
        conf=0.25,      # 신뢰도 임계값 (필요에 따라 조절)
        device='cpu'    # GPU가 있다면 '0', CPU만 사용한다면 'cpu' 지정
    )

    # 최종 저장 경로 계산
    final_output_path = os.path.join(output_folder, "result_images")
    print(f"✅ 처리가 완료되었습니다!")
    print(f"📁 결과 저장 폴더: {os.path.abspath(final_output_path)}")

if __name__ == "__main__":
    # --- 설정 영역 ---
    MODEL_FILE = "/Users/seongjun/Documents/국민대 자율주행/best_records_0724.pt"                     # 다운로드 받은 가중치 파일 이름
    INPUT_DIR = "/Users/seongjun/Documents/국민대 자율주행/cone"  # 예: C:/my_images 또는 ./images
    OUTPUT_DIR = "./yolo_output"                # 결과가 저장될 상위 폴더 이름
    
    # 실행
    run_yolo_inference(MODEL_FILE, INPUT_DIR, OUTPUT_DIR)