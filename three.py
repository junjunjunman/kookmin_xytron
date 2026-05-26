# -*- coding: utf-8 -*-
import cv2
import numpy as np

def three_mission(cv_image):
    """
    3구 신호등 미션 수행 모듈 (독립 함수 구조)
    입력: 메인 노드(track_drive.py)로부터 전달받은 원시 BGR 이미지
    반환: (angle, speed, status) 튜플 형태
    """
    if cv_image is None:
        return 0.0, 0.0, "wait"

    # 원본 이미지 보호를 위한 복사본 사용
    frame = cv_image.copy()

    # 1. 사용자가 test_three.py에서 검증 완료한 픽셀 좌표 기반 ROI 슬라이싱
    # [Y_시작:Y_끝, X_시작:X_끝] 구조 유지
    roi = frame[87:144, 336:393]

    # 2. 조명 변화에 강건한 HSV 색 공간 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 3. 사용자가 지정한 초록색 불빛의 HSV 검출 범위
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # 4. 범위 안의 픽셀만 흰색(255)으로 걸러내는 마스크 생성
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 5. 마스크 내부의 흰색 픽셀(녹색 신호) 개수 카운트
    green_pixel_count = cv2.countNonZero(mask)

    # =======================================================
    # [실시간 디버깅 창] 메인 코드 구동 중에도 실시간 확인 창 유지
    # =======================================================
    cv2.imshow("Traffic Light ROI", roi)
    cv2.imshow("Green Filter Mask", mask)
    cv2.waitKey(1)

    # 6. 사용자가 지정한 기준 픽셀 수(Threshold = 300) 기반 의사결정 및 리턴값 매핑
    if green_pixel_count > 300:
        # 터미널 창 가시성을 위해 print 문으로 디버깅 로그 제공
        print(f"🟢 [three.py] 초록불 확인! 픽셀 카운트: {green_pixel_count}")
        angle = 0.0
        speed = 15.0            # 다음 스테이지인 라바콘 구간으로 진입하기 위한 주행 속도 인가
        status = "greenlight"   # 메인 FSM을 STATE_CONE으로 천이시키는 트리거 문자열
    else:
        if green_pixel_count > 0:
            print(f"🔴 [three.py] 대기 중... (초록색 픽셀 수: {green_pixel_count}/300)")
        angle = 0.0
        speed = 0.0             # 신호 대기 중이므로 차량 정지 상태 유지
        status = "wait"         # 메인 FSM이 STATE_THREE_LIGHT를 유지하도록 세팅

    return angle, speed, status