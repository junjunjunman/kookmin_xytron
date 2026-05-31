#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def three_mission(cv_image):
    """
    정면 카메라의 이미지 특정 ROI 영역에서 3구 신호등의 녹색 판별 처리 [cite: 1014]
    
    :param cv_image: 카메라 원본 BGR 이미지
    :return: (angle, speed, status) 
    """
    # 1. 초기 반환값 설정 (기본 대기 상태)
    angle = 0.0
    speed = 0.0
    status = "wait"

    if cv_image is None:
        print("⚠️ [three_mission] 이미지가 수신되지 않았습니다.")
        return angle, speed, status

    # 2. 사용자가 마우스로 획득한 픽셀 좌표 기반 ROI 지정 [cite: 838]
    # Numpy Array 슬라이싱 구조: [Y_시작:Y_끝, X_시작:X_끝]
    roi = cv_image[87:144, 336:393]

    # 3. 조명 변화에 강건한 HSV 색 공간 변환 [cite: 838]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 4. 초록색(파란불) 불빛의 HSV 검출 범위 정의 [cite: 839]
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # 5. 범위 안의 픽셀만 걸러내는 마스크 생성 및 개수 카운트 [cite: 839]
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixel_count = cv2.countNonZero(mask)

    # =======================================================
    # [실시간 디버깅 창] 신호가 잘 잡히는지 눈으로 확인하는 뷰어
    # =======================================================
    # cv2.imshow("Traffic Light ROI", roi)
    # cv2.imshow("Green Filter Mask", mask)
    # cv2.waitKey(1)

    # 6. 기준 픽셀 수(Threshold) 초과 시 출발 신호 및 속도 발생 [cite: 841]
    if green_pixel_count > 300:
        print(f"🟢 [신호등 내부] 초록불 확인! 픽셀 카운트: {green_pixel_count} -> 차량 출발!")
        speed = 5.0   # 요청하신 출발 속도
        status = "greenlight" # 메인 상태 머신이 STATE_CONE으로 넘어가게 하는 트리거 
    else:
        # 노이즈를 줄이기 위해 픽셀이 0보다 클 때만 상태 로그 출력 [cite: 841]
        if green_pixel_count > 0:
            print(f"🔴 [신호등 내부] 빨간불/노란불 상태 (초록색 픽셀 수: {green_pixel_count}/300) -> 대기 중")

    # 메인 루프 규격에 맞춰 조향각, 속도, 상태 플래그 반환
    return angle, speed, status