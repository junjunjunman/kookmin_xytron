import cv2
import numpy as np
import math

# 전역 변수
points = []
scale_m_per_pixel = None  # 1픽셀당 몇 미터인지 저장
img_copy = None

def mouse_callback(event, x, y, flags, param):
    global points, scale_m_per_pixel, img_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_copy, (x, y), 4, (255, 0, 0), -1)
        
        if len(points) == 2:
            # 두 점을 선으로 연결
            pt1, pt2 = points[0], points[1]
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 2)
            
            # 픽셀 거리 계산
            pixel_dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            
            # 중앙 위치 (텍스트 출력용)
            mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
            
            # 스케일이 설정되지 않았다면 캘리브레이션 (0.11m 기준)
            if scale_m_per_pixel is None:
                if pixel_dist > 0:
                    scale_m_per_pixel = 0.11 / pixel_dist
                    print(f"[기준 설정 완료] 픽셀 거리: {pixel_dist:.2f}px = 0.11m")
                    print(f"[Scale] 1픽셀 = {scale_m_per_pixel:.5f}m")
                    cv2.putText(img_copy, "Ref: 0.11m", (mid_x, mid_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    print("경고: 같은 점을 클릭했습니다. 다시 클릭하세요.")
                    points.pop()
                    return
            # 스케일이 설정되어 있다면 실제 거리 측정
            else:
                real_dist = pixel_dist * scale_m_per_pixel
                print(f"[측정 완료] 두 점 사이의 실제 거리: {real_dist:.3f} m")
                cv2.putText(img_copy, f"{real_dist:.2f}m", (mid_x, mid_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
            points = [] # 다음 2개의 점을 받기 위해 초기화
            
        cv2.imshow("LiDAR Distance Calculator", img_copy)

def main():
    global img_copy, points, scale_m_per_pixel
    
    print("프로그램을 시작합니다...")

    # 1. 경로 설정 (한글 경로)
    img_path = "/Users/seongjun/Documents/국민대 자율주행/lidar_cone/lidar_0235.png"
    
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
    
    # 창 설정 (크기 조절 가능하도록 WINDOW_NORMAL 추가)
    cv2.namedWindow("LiDAR Distance Calculator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("LiDAR Distance Calculator", mouse_callback)
    
    print("\n=== 라이다 실제 거리 측정 툴 ===")
    print("[1단계] 기준점 설정: 실제 거리가 0.11m인 두 점(예: 십자가와 빨간 점)을 순서대로 클릭하세요.")
    print("[2단계] 거리 측정: 알고 싶은 두 점을 클릭하면 거리가 계산됩니다.")
    print(" * 'r' 키: 초기화 (스케일 재설정)")
    print(" * 'q' 또는 'ESC' 키: 종료\n")

    # 맥 OS 화면 멈춤 방지용 강제 렌더링
    cv2.imshow("LiDAR Distance Calculator", img_copy)
    cv2.waitKey(1)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 'r' 키: 초기화
        if key == ord('r') or key == ord('R'):
            img_copy = img.copy()
            points = []
            scale_m_per_pixel = None
            cv2.imshow("LiDAR Distance Calculator", img_copy)
            print("화면이 초기화되었습니다. 다시 0.11m 기준이 될 두 점을 클릭하세요.")
            
        # 'q' 또는 'ESC': 종료
        elif key == ord('q') or key == 27:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()