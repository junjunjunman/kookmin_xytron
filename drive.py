# -*- coding: utf-8 -*-
import cv2
import numpy as np

def drive_mission(cv_image, prev_L, prev_M, prev_R):
    """
    기존 StraightTestNode의 핵심 로직을 모듈 형태로 변환
    반환값: (angle, speed, status, next_L, next_M, next_R)
    """
    frame = cv_image.copy()
    h, w, _ = frame.shape
    
    # 1. HSV 필터링 및 통합 마스크 생성
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_lane = cv2.bitwise_or(mask_yellow, mask_white)

    # 2. 3차선 교점 인식 (Look-ahead 65%)
    scan_y = int(h * 0.65)
    mid_x = w // 2
    white_indices = np.where(mask_lane[scan_y, :] == 255)[0]

    centers = []
    if len(white_indices) > 0:
        current_cluster = [white_indices[0]]
        for idx in white_indices[1:]:
            if idx - current_cluster[-1] > 20:
                if len(current_cluster) >= 2: centers.append(int(np.mean(current_cluster)))
                current_cluster = [idx]
            else: current_cluster.append(idx)
        if len(current_cluster) >= 2: centers.append(int(np.mean(current_cluster)))
    
    centers = [c for c in centers if int(w * 0.08) < c < int(w * 0.92)]
    ideal_lane_width = int(w * 0.48)
    half_width = ideal_lane_width // 2

    # 기준 좌표 매칭
    ref_L = prev_L if prev_L is not None else (mid_x - half_width)
    ref_M = prev_M if prev_M is not None else mid_x
    ref_R = prev_R if prev_R is not None else (mid_x + half_width)

    det_L, det_M, det_R = None, None, None

    # 비용 최소화 기반 차선 매칭 및 복구 로직
    if len(centers) >= 3:
        det_L, det_M, det_R = centers[0], centers[1], centers[2]
    elif len(centers) == 2:
        cost_LM = abs(centers[0] - ref_L) + abs(centers[1] - ref_M)
        cost_MR = abs(centers[0] - ref_M) + abs(centers[1] - ref_R)
        cost_LR = abs(centers[0] - ref_L) + abs(centers[1] - ref_R)
        
        min_cost = min(cost_LM, cost_MR, cost_LR)
        if min_cost == cost_LM:
            det_L, det_M = centers[0], centers[1]
        elif min_cost == cost_MR:
            det_M, det_R = centers[0], centers[1]
        else:
            det_L, det_R = centers[0], centers[1]
    elif len(centers) == 1:
        cost_L = abs(centers[0] - ref_L)
        cost_M = abs(centers[0] - ref_M)
        cost_R = abs(centers[0] - ref_R)
        
        min_cost = min(cost_L, cost_M, cost_R)
        if min_cost == cost_L:
            det_L = centers[0]
        elif min_cost == cost_M:
            det_M = centers[0]
        else:
            det_R = centers[0]

    # 기하학적 복구 알고리즘 (Reconstruction)
    if det_L is not None and det_M is not None and det_R is not None:
        final_L, final_M, final_R = det_L, det_M, det_R
    elif det_L is not None and det_M is not None:
        final_L, final_M = det_L, det_M
        final_R = final_M + half_width
    elif det_M is not None and det_R is not None:
        final_M, final_R = det_M, det_R
        final_L = final_M - half_width
    elif det_L is not None and det_R is not None:
        final_L, final_R = det_L, det_R
        final_M = (final_L + final_R) // 2
    elif det_L is not None:
        final_L = det_L
        final_M = final_L + half_width
        final_R = final_L + ideal_lane_width
    elif det_M is not None:
        final_M = det_M
        final_L = final_M - half_width
        final_R = final_M + half_width
    elif det_R is not None:
        final_R = det_R
        final_M = final_R - half_width
        final_L = final_R - ideal_lane_width
    else:
        final_L, final_M, final_R = ref_L, ref_M, ref_R

    # 3. 제어 및 동적 감속 알고리즘
    lane_shift_rate = 0
    if prev_L is not None:
        lane_shift_rate = max(abs(final_L - prev_L), abs(final_M - prev_M), abs(final_R - prev_R))
    
    is_lane_corrupted = (final_L >= final_M) or (final_M >= final_R)
    
    target_x = (final_M + final_R) // 2
    error = target_x - mid_x
    
    if not is_lane_corrupted:
        adaptive_kp = 0.4 + (lane_shift_rate * 0.02)
        angle = float(error * min(adaptive_kp, 1.2))
        speed = max(5.5, 15.5 - (abs(angle) * 0.32) - (lane_shift_rate * 0.15))
        status = "driving"
    else:
        angle = 30.0 if (np.sum(mask_lane[scan_y:, :mid_x]) > np.sum(mask_lane[scan_y:, mid_x:])) else -30.0
        speed = 4.0
        status = "driving"

    # 4. ROI 기반 정지선/체크보드 인터록
    roi = mask_white[int(h*0.82):int(h*0.92), int(w*0.25):int(w*0.75)]
    if np.sum(roi == 255) > 3000:
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if len(valid_contours) >= 3:
            status = "driving" # 체크보드 스루
        elif len(valid_contours) > 0:
            status = "stop_line_detected" # 신호등 정지
            speed, angle = 0.0, 0.0

    return angle, speed, status, final_L, final_M, final_R