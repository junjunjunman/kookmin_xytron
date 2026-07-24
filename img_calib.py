import cv2
import numpy as np
import os
import glob

def process_images():
    input_dir = "img"
    output_dir = "img_calib"
    os.makedirs(output_dir, exist_ok=True)

    # 캘리브레이션 파라미터 로드
    data = np.load("camera_params.npz")
    camera_matrix = data["mtx"]
    dist_coeffs = data["dist"]

    # 이미지 탐색
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 왜곡 보정
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (w, h),
            0,
            (w, h)
        )

        undistorted = cv2.undistort(
            img,
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix
        )

        # 검은 테두리 제거
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y+rh, x:x+rw]

        # 같은 이름으로 저장
        fname = os.path.basename(img_path)
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, undistorted)

        print(f"[{fname}] 저장 완료")

    print(f"\n작업 완료! 결과는 '{output_dir}/' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    process_images()