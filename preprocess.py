import cv2
import numpy as np
import os
import glob

# ==========================================
# 파라미터 설정
# ==========================================
INPUT_DIR = "img"
OUTPUT_DIR = "output_results"

# 파이프라인 설정 (원하는 순서대로 입력)
# 옵션: 'gray', 'blur', 'canny', 'sobel', 'laplacian', 'adaptive', 'hough'
PIPELINE = ['gray', 'blur', 'canny']

# --- 세부 파라미터 ---
# Gaussian Blur
BLUR_KSIZE = (5, 5)

# Canny Edge
CANNY_TH1 = 50
CANNY_TH2 = 150

# Sobel / Laplacian
SOBEL_KSIZE = 3

# Adaptive Threshold
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2

# Hough Line Transform (HoughLinesP)
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LEN = 100
HOUGH_MAX_LINE_GAP = 50
# ==========================================

def process_image(img, pipeline):
    res = img.copy()
    
    for step in pipeline:
        if step == 'gray':
            if len(res.shape) == 3:
                res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                
        elif step == 'blur':
            res = cv2.GaussianBlur(res, BLUR_KSIZE, 0)
            
        elif step == 'canny':
            res = cv2.Canny(res, CANNY_TH1, CANNY_TH2)
            
        elif step == 'sobel':
            sobelx = cv2.Sobel(res, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
            sobely = cv2.Sobel(res, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
            res = cv2.magnitude(sobelx, sobely)
            res = np.uint8(np.clip(res, 0, 255))
            
        elif step == 'laplacian':
            res = cv2.Laplacian(res, cv2.CV_64F)
            res = cv2.convertScaleAbs(res)
            
        elif step == 'adaptive':
            if len(res.shape) == 3:
                res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
            
        elif step == 'hough':
            lines = cv2.HoughLinesP(res, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                                    minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP)
            
            # 컬러 이미지 변환 후 빨간 선 그리기
            if len(res.shape) == 2:
                res_color = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
            else:
                res_color = res.copy()
                
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line.flatten()
                    cv2.line(res_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            res = res_color
            
    return res

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    search_pattern = os.path.join(INPUT_DIR, "lane (*).png")
    img_paths = glob.glob(search_pattern)
    
    if not img_paths:
        print(f"'{INPUT_DIR}' 폴더에서 이미지를 찾을 수 없습니다.")
        return

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            continue
            
        # 파이프라인 적용
        result = process_image(img, PIPELINE)
        
        # 결과 저장
        filename = os.path.basename(path)
        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, result)
        print(f"저장 완료: {out_path}")

if __name__ == "__main__":
    main()