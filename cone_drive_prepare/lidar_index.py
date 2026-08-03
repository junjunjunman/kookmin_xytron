import cv2
import numpy as np
import math

# 전역 변수
center_pt = None
saved_markers = []  # 클릭해서 고정해둔 인덱스 정보를 저장하는 리스트
img_original = None
img_display = None
TOTAL_INDICES = 504  # 총 라이다 인덱스 개수 (504등분)

def get_index_and_angle(cx, cy, x, y):
    """중심점(cx, cy)과 대상점(x, y)을 받아 각도와 인덱스를 반환하는 함수"""
    # OpenCV는 화면 아래로 갈수록 y값이 커집니다.
    dx = x - cx
    dy = y - cy
    
    # 남쪽(dy > 0, dx = 0)일 때 0도가 되도록 atan2(dx, dy) 사용
    angle_rad = math.atan2(dx, dy)
    
    # 음수 각도를 0~360도로 변환 (남쪽=0도, 동쪽=90도, 북쪽=180도, 서쪽=270도)
    if angle_rad < 0:
        angle_rad += 2 * math.pi
    
    angle_deg = math.degrees(angle_rad)
    
    # 인덱스 계산 (0~360도를 1~504 범위로 매핑)
    index = int((angle_deg / 360.0) * TOTAL_INDICES) + 1
    if index > TOTAL_INDICES:
        index = 1
        
    return index, angle_deg

def mouse_callback(event, x, y, flags, param):
    global center_pt, saved_markers, img_display, img_original

    # [1] 마우스 좌클릭 이벤트 (중심점 설정 OR 인덱스 고정)
    if event == cv2.EVENT_LBUTTONDOWN:
        if center_pt is None:
            # 첫 번째 클릭: 중심점(로봇 센터) 설정
            center_pt = (x, y)
            img_display = img_original.copy()
            cv2.circle(img_display, center_pt, 5, (0, 0, 255), -1)
            cv2.imshow("Lidar Index Finder", img_display)
            print(f"[설정 완료] 중심점: {center_pt}")
            print("👉 이제 원하는 곳으로 마우스를 옮긴 뒤 클릭하면 인덱스가 화면에 고정됩니다!")
        else:
            # 두 번째 이후 클릭: 해당 위치의 인덱스 화면에 고정(저장)
            cx, cy = center_pt
            idx, ang = get_index_and_angle(cx, cy, x, y)
            saved_markers.append((x, y, idx, ang))
            print(f"[인덱스 핀 설정] Idx: {idx} / {ang:.1f}도")

    # [2] 중심점이 설정된 상태에서 마우스를 움직일 때 (실시간 미리보기)
    elif event == cv2.EVENT_MOUSEMOVE:
        if center_pt is not None:
            img_display = img_original.copy()
            cx, cy = center_pt
            
            # 기준선 그리기 (십자선 가이드)
            cv2.line(img_display, (cx, 0), (cx, img_display.shape[0]), (200, 200, 200), 1)
            cv2.line(img_display, (0, cy), (img_display.shape[1], cy), (200, 200, 200), 1)
            cv2.circle(img_display, (cx, cy), 4, (0, 0, 255), -1)
            
            # 고정(저장)된 마커들 모두 그리기 (주황색 선)
            for mx, my, midx, mang in saved_markers:
                cv2.line(img_display, (cx, cy), (mx, my), (0, 150, 255), 2)
                cv2.circle(img_display, (mx, my), 4, (0, 150, 255), -1)
                cv2.putText(img_display, f"Idx: {midx}", (mx + 10, my - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
            
            # 마우스 커서까지 실시간 선 연결 (초록색 선)
            cv2.line(img_display, (cx, cy), (x, y), (0, 255, 0), 1)
            idx, ang = get_index_and_angle(cx, cy, x, y)
            
            # 실시간 텍스트 출력
            text1 = f"Idx: {idx} / {TOTAL_INDICES}"
            text2 = f"{ang:.1f} deg"
            cv2.putText(img_display, text1, (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img_display, text2, (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            
            cv2.imshow("Lidar Index Finder", img_display)

def main():
    global img_original, img_display, center_pt, saved_markers
    
    print("프로그램을 시작합니다...")

    # 1. 파일 경로 (한글 경로 지원)
    img_path = "/Users/seongjun/Documents/국민대 자율주행/lidar_cone/lidar_0235.png"
    
    img_array = np.fromfile(img_path, np.uint8)
    if img_array.size == 0:
        print(f"🚨 Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요!\n{img_path}")
        return
        
    img_original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_original is None:
        print("🚨 Error: 이미지를 디코딩할 수 없습니다.")
        return
        
    img_display = img_original.copy()
    
    cv2.namedWindow("Lidar Index Finder", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Lidar Index Finder", mouse_callback)
    
    print("\n=== 라이다 504등분 인덱스 탐색기 (남쪽 기준) ===")
    print("1. 마우스 좌클릭(1회): 0도(인덱스 1) 기준이 될 중심점 설정")
    print("2. 마우스 이동: 실시간 인덱스 표시 (초록선)")
    print("3. 마우스 좌클릭(2회 이상): 원하는 인덱스 화면에 핀 꽂기 (주황선)")
    print("4. 'r' 키: 화면 초기화 (모든 핀 제거 및 중심점 재설정)")
    print("5. 'q' 또는 'ESC' 키: 종료\n")

    cv2.imshow("Lidar Index Finder", img_display)
    cv2.waitKey(1)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 'r' 키: 초기화
        if key == ord('r') or key == ord('R'):
            img_display = img_original.copy()
            center_pt = None
            saved_markers = []  # 저장된 마커들도 모두 날림
            cv2.imshow("Lidar Index Finder", img_display)
            print("초기화되었습니다. 다시 중심점을 클릭하세요.")
            
        # 'q' 또는 'ESC': 종료
        elif key == ord('q') or key == 27:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()