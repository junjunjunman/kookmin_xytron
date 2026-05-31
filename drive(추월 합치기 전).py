#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

# 전역 상태 변수 (FSM 내부 상태 유지용)
is_stopped_at_line = False
ignore_stopline_counter = 0
is_school_zone = False
locked_obs = None
last_normal_angle = 0.0
last_normal_speed = 15.0

def drive_mission(cv_image, prev_L, prev_M, prev_R, lidar_ranges=None):
    global is_stopped_at_line, ignore_stopline_counter
    global is_school_zone, locked_obs, last_normal_angle, last_normal_speed
    
    h, w, _ = cv_image.shape
    display_img = cv_image.copy()
    out_status = "driving"

    # 1. 정지선 무시 쿨다운 갱신
    if ignore_stopline_counter > 0:
        ignore_stopline_counter -= 1

    # 2. 이미지 색상 필터링 (HSV)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_lane = cv2.bitwise_or(mask_yellow, mask_white)

    # 3. 차선 인식 알고리즘 (Look-ahead 스캔)
    scan_y = int(h * 0.65)
    mid_x = w // 2
    scan_line_row = mask_lane[scan_y, :]
    white_indices = np.where(scan_line_row == 255)[0]

    centers = []
    if len(white_indices) > 0:
        current_cluster = [white_indices[0]]
        for idx in white_indices[1:]:
            if idx - current_cluster[-1] > 20:
                if len(current_cluster) >= 2:  
                    centers.append(int(np.mean(current_cluster)))
                current_cluster = [idx]
            else:
                current_cluster.append(idx)
        if len(current_cluster) >= 2:
            centers.append(int(np.mean(current_cluster)))

    centers = [c for c in centers if int(w * 0.08) < c < int(w * 0.92)]
    ideal_lane_width = int(w * 0.48)
    half_width = ideal_lane_width // 2

    ref_L = prev_L if prev_L is not None else (mid_x - half_width)
    ref_M = prev_M if prev_M is not None else mid_x
    ref_R = prev_R if prev_R is not None else (mid_x + half_width)

    det_L, det_M, det_R = None, None, None

    # 차선 매칭 및 기하학적 복구
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
        cost_L, cost_M, cost_R = abs(centers[0] - ref_L), abs(centers[0] - ref_M), abs(centers[0] - ref_R)
        min_cost = min(cost_L, cost_M, cost_R)
        if min_cost == cost_L: det_L = centers[0]
        elif min_cost == cost_M: det_M = centers[0]
        else: det_R = centers[0]

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

    lane_shift_rate = 0
    if prev_L is not None and prev_M is not None and prev_R is not None:
        lane_shift_rate = max(abs(final_L - prev_L), abs(final_M - prev_M), abs(final_R - prev_R))

    is_lane_corrupted = False
    if (final_L >= final_M) or (final_M >= final_R) or (final_R - final_L > ideal_lane_width * 1.5) or (final_R - final_L < ideal_lane_width * 0.5):
        is_lane_corrupted = True

    base_speed = 25.0
    base_kp = 0.4
    target_x = (final_L + final_R) // 2
    error = target_x - mid_x

    # 4. 일반 주행 제어량 연산 (스쿨존 감지 포함)
    if not is_lane_corrupted:
        adaptive_kp = base_kp
        if lane_shift_rate > 10 or abs(error) > int(w * 0.05):
            adaptive_kp += (lane_shift_rate * 0.02) + (abs(error) * 0.004)
            adaptive_kp = min(adaptive_kp, 1.2)

        calc_angle = float(error * adaptive_kp)
        calc_speed = base_speed - (abs(calc_angle) * 0.32) - (lane_shift_rate * 0.15)
        
        if abs(calc_angle) > 15.0 or lane_shift_rate > 12:
            calc_speed = min(calc_speed, 8.0)
        calc_speed = max(10.0, calc_speed)
        
        # Yellow line 인식 스쿨존 모드 진입/탈출 로직 (98% Scan Line 적용)
        check_window = 15
        is_L_yellow = np.any(mask_yellow[scan_y, max(0, final_L - check_window):min(w, final_L + check_window)] == 255)
        is_R_yellow = np.any(mask_yellow[scan_y, max(0, final_R - check_window):min(w, final_R + check_window)] == 255)
        
        if is_L_yellow and is_R_yellow:
            is_school_zone = True
            
        if is_school_zone:
            scan_y_99 = int(h * 0.98) # 98% 하단 라인 검사
            white_pixels_at_99 = np.sum(mask_white[scan_y_99, :] == 255)
            if white_pixels_at_99 > 15: # 흰색 실선 복귀 감지 시 탈출
                is_school_zone = False
                
        if is_school_zone:
            calc_speed = 5.0
            
        next_L, next_M, next_R = final_L, final_M, final_R
    else:
        calc_speed = 4.0  
        left_weight = np.sum(mask_lane[scan_y:, :mid_x] == 255)
        right_weight = np.sum(mask_lane[scan_y:, mid_x:] == 255)
        if right_weight > left_weight * 1.3: calc_angle = -30.0  
        elif left_weight > right_weight * 1.3: calc_angle = 30.0   
        else:
            calc_angle = float(error * base_kp)
            calc_speed = 5.5
        next_L, next_M, next_R = ref_L, ref_M, ref_R

    # 5. 전방 라이다 장애물(보행자 등) 탐지
    person_detected = False
    if lidar_ranges is not None and len(lidar_ranges) > 0:
        front_idx = list(range(0, 70)) + list(range(330, 360))
        valid_front = np.array(lidar_ranges)[front_idx]
        valid_front = valid_front[valid_front > 0.1]
        if len(valid_front) > 0 and np.min(valid_front) < 1.5:
            person_detected = True

    # 6. 정지선 감지
    roi_x_start, roi_x_end = int(w * 0.25), int(w * 0.75)
    roi_y_start, roi_y_end = int(h * 0.70), int(h * 0.80)
    stop_line_roi = mask_white[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    white_ratio = np.sum(stop_line_roi == 255) / ((roi_x_end - roi_x_start) * (roi_y_end - roi_y_start))

    if not is_stopped_at_line and ignore_stopline_counter == 0 and not person_detected:
        is_checkerboard = False
        if white_ratio > 0.15: 
            contours, _ = cv2.findContours(stop_line_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_blobs = [c for c in contours if cv2.contourArea(c) > 50]
            if len(valid_blobs) >= 3:
                is_checkerboard = True

        if not is_checkerboard and white_ratio >= 0.70:
            is_stopped_at_line = True
            locked_obs = None # 신규 정지선 진입 시 라이다 장애물 래치 초기화

    # 7. 4구 신호등 및 교차로 장애물 의사결정 (check_light.py 로직)
    if is_stopped_at_line:
        calc_speed = 0.0
        calc_angle = 0.0
        
        roi_x1, roi_x2 = 86, 428
        roi_y1, roi_y2 = 31, 153
        roi = cv_image[roi_y1:roi_y2, roi_x1:roi_x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        mask_red1 = cv2.inRange(hsv_roi, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv_roi, np.array([170, 50, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow_tl = cv2.inRange(hsv_roi, np.array([15, 80, 80]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(hsv_roi, np.array([40, 50, 50]), np.array([90, 255, 255]))

        red_cnt = cv2.countNonZero(mask_red)
        yellow_cnt = cv2.countNonZero(mask_yellow_tl)
        green_cnt = cv2.countNonZero(mask_green)

        threshold = 1000 
        is_red_on = red_cnt > threshold
        is_yellow_on = yellow_cnt > threshold
        is_green_on = green_cnt > threshold

        if is_red_on and is_green_on: light_status = "Left (Red+Green)"
        elif is_yellow_on: light_status = "Yellow"
        elif is_red_on and not is_green_on: light_status = "Stop (Red)"
        elif is_green_on and not is_red_on: light_status = "Straight (Green)"
        else: light_status = "Unknown"

        # 좌측 장애물 (경찰차 등) 판정: 20~80도 범위, 5m 이내
        current_obs = False
        if lidar_ranges is not None and len(lidar_ranges) > 0:
            for i, d in enumerate(lidar_ranges):
                if not math.isfinite(d) or d <= 0.05: continue
                if 20 <= i <= 80 and d < 5.0:
                    current_obs = True
                    break
                    
        # 래치 (한번이라도 포착되면 True 유지)
        if locked_obs is None: locked_obs = current_obs
        elif current_obs: locked_obs = True
        obs = locked_obs

        # Decision 로직
        final_decision = "Waiting..."
        if light_status in ["Stop (Red)", "Yellow"]:
            final_decision = "Wait for Left -> TURN LEFT" if not obs else "Wait for Straight -> GO STRAIGHT"
        elif light_status == "Left (Red+Green)":
            final_decision = "ACTION: TURN LEFT NOW" if not obs else "Wait for Straight -> GO STRAIGHT"
        elif light_status == "Straight (Green)":
            final_decision = "Wait for Left -> TURN LEFT" if not obs else "ACTION: GO STRAIGHT NOW"

        # 주행 허가 시 정지선 Lock 해제 및 out_status 발송
        if "ACTION: TURN LEFT NOW" in final_decision:
            is_stopped_at_line = False
            ignore_stopline_counter = 50
            out_status = "left"
        elif "ACTION: GO STRAIGHT NOW" in final_decision:
            is_stopped_at_line = False
            ignore_stopline_counter = 50
            out_status = "driving"

    # 8. 최종 조향각/속도 할당 
    if person_detected and not is_stopped_at_line:
        angle = last_normal_angle
        speed = 0.0
    else:
        angle = calc_angle
        speed = calc_speed
        if not is_stopped_at_line:
            last_normal_angle = angle
            last_normal_speed = speed
            
    # 디버깅 시각화 창 (필요 시 제거 가능)
    cv2.imshow("Drive Tracking Module", display_img) 
    cv2.waitKey(1)

    return angle, float(speed), out_status, next_L, next_M, next_R