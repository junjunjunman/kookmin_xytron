import cv2
import numpy as np

# 전역 변수 초기화
current_points = []
polygons = []
img_copy = None

def mouse_callback(event, x, y, flags, param):
    global current_points, img_copy
    
    # 마우스 좌클릭 시 점 추가
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
        
        # 점이 2개 이상이면 선으로 연결
        if len(current_points) > 1:
            cv2.line(img_copy, current_points[-2], current_points[-1], (0, 255, 0), 2)
            
        cv2.imshow("YOLO Seg Area Calculator", img_copy)

def main():
    global img_copy, current_points, polygons
    
    print("프로그램을 시작합니다...")

    # 1. 경로 설정 (한글 경로 포함)
    img_path = "/Users/seongjun/Documents/국민대 자율주행/cone/cone_0235.png"
    
    # 2. 한글 경로를 지원하는 이미지 불러오기 방식
    img_array = np.fromfile(img_path, np.uint8)
    if img_array.size == 0:
        print(f"🚨 Error: 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요!\n입력된 경로: {img_path}")
        return
        
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print("🚨 Error: 파일은 찾았으나 이미지로 디코딩할 수 없습니다.")
        return
        
    img_copy = img.copy()
    
    cv2.namedWindow("YOLO Seg Area Calculator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLO Seg Area Calculator", mouse_callback)
    
    print("\n=== 다각형 넓이 측정 툴 ===")
    print("1. 마우스 좌클릭: 다각형의 꼭짓점 추가")
    print("2. 'c' 키: 다각형 닫기 및 넓이 계산")
    print("3. 'r' 키: 화면 초기화 (모두 지우기)")
    print("4. 'q' 또는 'ESC' 키: 프로그램 종료\n")

    cv2.imshow("YOLO Seg Area Calculator", img_copy)
    cv2.waitKey(1)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 'c' 키: 다각형 닫기 및 면적 계산
        if key == ord('c') or key == ord('C'):
            if len(current_points) >= 3:
                cv2.line(img_copy, current_points[-1], current_points[0], (0, 255, 0), 2)
                
                contour = np.array(current_points, dtype=np.int32)
                area = cv2.contourArea(contour)
                polygons.append((contour, area))
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(img_copy, f"Area: {area:.1f}", (cX - 40, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                print(f"[다각형 {len(polygons)}] 픽셀 면적(Area): {area:.1f}")
                current_points = [] 
                cv2.imshow("YOLO Seg Area Calculator", img_copy)
            else:
                print("경고: 다각형을 닫으려면 최소 3개의 점이 필요합니다.")
                
        # 'r' 키: 초기화
        elif key == ord('r') or key == ord('R'):
            img_copy = img.copy()
            current_points = []
            polygons = []
            cv2.imshow("YOLO Seg Area Calculator", img_copy)
            print("화면이 초기화되었습니다.")
            
        # 'q' 또는 'ESC': 종료
        elif key == ord('q') or key == 27:
            break
            
    cv2.destroyAllWindows()

# [수정된 부분] 들여쓰기가 맨 왼쪽으로 붙어있어야 합니다.
if __name__ == "__main__":
    main()