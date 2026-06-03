#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import time

# ==========================================
# 전역 상태 변수
# ==========================================
_is_stopped_at_line = False
_locked_obs = None
_mission_status = "drive"
_ignore_stopline_counter = 0

# 보행자 상태
pedestrian_stop = False
was_pedestrian_stop = False
saved_speed = 0.0
saved_angle = 0.0
prev_normal_speed = 10.0
prev_normal_angle = 0.0

# 추월(Overtake) 상태 
overtake_state = 0 
overtake_direction = ""
overtake_step_start_time = 0.0  
passing_clear_time = 0.0

# Yellow Zone 타이머
_is_yellow_zone = False  
_yellow_exit_time = 0.0  

# HSV 마스크 임계값
lower_yellow = np.array([29, 50, 250])
upper_yellow = np.array([31, 255, 255])
lower_white = np.array([0, 0, 250])
upper_white = np.array([0, 20, 255])

# ==========================================
# Yellow Line Checking 함수
# ==========================================
def check_yellow_line(scan_y, mask_lane, mask_yellow, w, check_w=20):
    row = mask_lane[scan_y, :]
    white_indices = np.where(row == 255)[0]
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

    centers = [c for c in centers if int(w * 0.05) < c < int(w * 0.95)]
    
    yellow_centers = []
    for c in centers:
        if np.any(mask_yellow[scan_y, max(0, c - check_w):min(w, c + check_w)] == 255):
            yellow_centers.append(c)
            
    return len(yellow_centers) >= 3, yellow_centers

# ==========================================
# track_drive.py 에서 호출될 메인함수
# ==========================================
def drive_mission(cv_image, prev_L, prev_M, prev_R, lidar_ranges):
    global _is_stopped_at_line, _locked_obs, _mission_status, _ignore_stopline_counter
    global pedestrian_stop, was_pedestrian_stop, saved_speed, saved_angle
    global prev_normal_speed, prev_normal_angle
    global overtake_state, overtake_direction, overtake_step_start_time, passing_clear_time
    global _is_yellow_zone, _yellow_exit_time

    current_time = time.time()
    
    # 기본 상태 초기화
    _mission_status = "drive"
    
    # ==========================================================
    # 0. 라이다 데이터 분석
    # ==========================================================
    pedestrian_count = 0
    obs_intersection_count = 0
    
    overtake_trigger = 0
    overtake_left_0_60 = 0
    overtake_right_300_360 = 0
    passing_left_count = 0
    passing_right_count = 0

    if lidar_ranges is not None and len(lidar_ranges) > 0:
        for i, d in enumerate(lidar_ranges):
            if not math.isfinite(d) or d <= 0.05: continue
            
            # 교차로 장애물 판별
            if 30 <= i <= 60 and d < 6.0:
                obs_intersection_count += 1
                
            # 보행자 판별 (시야각 70도, 거리 3.0)
            if (0 <= i <= 70 or 290 <= i <= 359) and d < 3.0:
                pedestrian_count += 1
                
            # 추월 트리거 판별
            if 4.0 <= d <= 8.5:
                if 0 <= i <= 60 or 300 <= i <= 359:
                    overtake_trigger += 1
                
            # 추월 방향 판별
            if d <= 5.0:
                if 0 <= i <= 60:
                    overtake_left_0_60 += 1
                elif 300 <= i <= 359:
                    overtake_right_300_360 += 1
                
            # 추월 여부 판별
            if d < 3.5:
                if 30 <= i <= 90: passing_left_count += 1
                if 270 <= i <= 330: passing_right_count += 1

    # 보행자 발견 기준
    if 10 <= pedestrian_count <= 40 and overtake_state == 0:
        pedestrian_stop = True
    else:
        pedestrian_stop = False

    # ==========================================================
    # 1. 이미지 처리
    # ==========================================================
    h, w, _ = cv_image.shape
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_lane = cv2.bitwise_or(mask_yellow, mask_white)
    
    display_img = cv2.bitwise_and(cv_image, cv_image, mask=mask_lane)

    # ==========================================================
    # 2. 정지선 탐지
    # ==========================================================
    roi_stop_x1, roi_stop_x2 = 212, 435
    roi_stop_y1, roi_stop_y2 = 330, 377
    white_ratio = 0.0
    
    if _ignore_stopline_counter > 0:
        _ignore_stopline_counter -= 1
    else:
        stop_roi = mask_white[roi_stop_y1:roi_stop_y2, roi_stop_x1:roi_stop_x2]
        total_pixels = stop_roi.size
        white_pixels = np.sum(stop_roi == 255)
        white_ratio = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        if white_ratio >= 60.0 and not _is_stopped_at_line and overtake_state in [0, 0.5] and not pedestrian_stop:
            _is_stopped_at_line = True
            _locked_obs = None  
            print(f"정지선 인식! (비율: {white_ratio:.1f}%)")

    # ==========================================================
    # 3(1). Scan Line(65%)과 차선 교점 찾기
    # ==========================================================
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

    if len(centers) >= 3:
        det_L, det_M, det_R = centers[0], centers[1], centers[2]
    elif len(centers) == 2:
        cost_LM = abs(centers[0] - ref_L) + abs(centers[1] - ref_M)
        cost_MR = abs(centers[0] - ref_M) + abs(centers[1] - ref_R)
        cost_LR = abs(centers[0] - ref_L) + abs(centers[1] - ref_R)
        min_cost = min(cost_LM, cost_MR, cost_LR)
        if min_cost == cost_LM: det_L, det_M = centers[0], centers[1]
        elif min_cost == cost_MR: det_M, det_R = centers[0], centers[1]
        else: det_L, det_R = centers[0], centers[1]
    elif len(centers) == 1:
        cost_L, cost_M, cost_R = abs(centers[0] - ref_L), abs(centers[0] - ref_M), abs(centers[0] - ref_R)
        min_cost = min(cost_L, cost_M, cost_R)
        if min_cost == cost_L: det_L = centers[0]
        elif min_cost == cost_M: det_M = centers[0]
        else: det_R = centers[0]

    # 결손 차선 위치 유추 복원
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

    # 급격한 교점 튐 방지
    if prev_M is not None:
        shift = final_M - prev_M
        if abs(shift) > 40:
            final_M = prev_M + (40 if shift > 0 else -40)
            final_L = final_M - half_width
            final_R = final_M + half_width

    # ==========================================================
    # 3(2). Scan Line(70%) & 2초마다 탐지
    # ==========================================================
    scan_y_70 = int(h * 0.70)
    is_yellow_70, yellow_pts_70 = check_yellow_line(scan_y_70, mask_lane, mask_yellow, w)

    if overtake_state == 0:
        if is_yellow_70:
            _is_yellow_zone = True
            _yellow_exit_time = 0.0  
        elif _is_yellow_zone:
            if _yellow_exit_time == 0.0:
                _yellow_exit_time = current_time
            elif (current_time - _yellow_exit_time) > 2.0:
                _is_yellow_zone = False
                _yellow_exit_time = 0.0

    # ==========================================================
    # 4. 추월 트리거 ~ 추월 방향까지 판단
    # ==========================================================
    if not _is_stopped_at_line and not pedestrian_stop:
        if overtake_state == 0:
            if overtake_trigger >= 25:
                overtake_state = 0.5
                print(f"[OVERTAKE] DETECTED 전방 {overtake_trigger}개.")

        elif overtake_state == 0.5:
            is_front_horizontal = False
            front_y_list = []
            
            if lidar_ranges is not None and len(lidar_ranges) > 0:
                for i, d in enumerate(lidar_ranges):
                    if (0 <= i <= 60 or 300 <= i <= 359) and math.isfinite(d) and 0.05 < d <= 8.5:
                        angle = math.radians(i - 90)
                        y = -d * math.sin(angle)
                        front_y_list.append(y)

            if len(front_y_list) >= 20:
                y_std = np.std(front_y_list)
                if y_std < 0.3:  
                    is_front_horizontal = True

            if is_front_horizontal:
                if overtake_left_0_60 >= 20 and overtake_left_0_60 > overtake_right_300_360:
                    overtake_state = 1
                    overtake_direction = "RIGHT"
                    overtake_step_start_time = current_time
                    print("-> [OVERTAKE] RIGHT (2차선으로 이동)")
                    
                elif overtake_right_300_360 >= 20 and overtake_right_300_360 > overtake_left_0_60:
                    overtake_state = 1
                    overtake_direction = "LEFT"
                    overtake_step_start_time = current_time
                    print("->[OVERTAKE] LEFT (1차선으로 이동)")
            else:
                if overtake_trigger < 5 and overtake_left_0_60 < 5 and overtake_right_300_360 < 5:
                    overtake_state = 0
                    print("-> [OVERTAKE] 취소: 장애물 사라짐.")

    # ==========================================================
    # 5. 상태에 따른 조향각 및 속도 결정 (Adaptive P 제어)
    # ==========================================================
    lane_shift_rate = 0
    if prev_L is not None and prev_M is not None and prev_R is not None:
        shift_L = abs(final_L - prev_L)
        shift_M = abs(final_M - prev_M)
        shift_R = abs(final_R - prev_R)
        lane_shift_rate = max(shift_L, shift_M, shift_R)

    is_lane_corrupted = False
    if (final_L >= final_M) or (final_M >= final_R) or (final_R - final_L > ideal_lane_width * 1.5) or (final_R - final_L < ideal_lane_width * 0.5):
        is_lane_corrupted = True

    base_speed = 20.0
    base_kp = 0.4
    
    if overtake_state == 1:
        if overtake_direction == "RIGHT": target_x = (final_M + final_R) // 2
        else: target_x = (final_L + final_M) // 2
    elif overtake_state == 2:
        if overtake_direction == "RIGHT": target_x = (final_L + final_M) // 2
        else: target_x = (final_M + final_R) // 2
    else:
        target_x = (final_L + final_R) // 2

    error = target_x - mid_x
    
    status_text = f"DRIVING (Status: {_mission_status})"
    text_color = (0, 255, 0)
    calc_speed, calc_angle = 0.0, 0.0

    # ----------------------------------------------------------
    # [A] 교차로 신호 대기
    # ----------------------------------------------------------
    if _is_stopped_at_line:
        calc_speed = 0.0
        calc_angle = 0.0
        status_text = f"신호 기다리는 중... (Status: {_mission_status})"
        text_color = (0, 0, 255)
        
        roi_tl_x1, roi_tl_x2 = 80, 420
        roi_tl_y1, roi_tl_y2 = 60, 150
        roi = cv_image[roi_tl_y1:roi_tl_y2, roi_tl_x1:roi_tl_x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        mask_red1 = cv2.inRange(hsv_roi, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv_roi, np.array([170, 50, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow_tl = cv2.inRange(hsv_roi, np.array([15, 80, 80]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(hsv_roi, np.array([40, 50, 50]), np.array([90, 255, 255]))

        threshold = 1000 
        is_red_on = cv2.countNonZero(mask_red) > threshold
        is_yellow_on = cv2.countNonZero(mask_yellow_tl) > threshold
        is_green_on = cv2.countNonZero(mask_green) > threshold

        if is_red_on and is_green_on: light_status = "Left (Red+Green)"
        elif is_yellow_on: light_status = "Yellow"
        elif is_red_on and not is_green_on: light_status = "Stop (Red)"
        elif is_green_on and not is_red_on: light_status = "Straight (Green)"
        else: light_status = "Unknown"

        current_obs = False
        if obs_intersection_count >= 3:
            current_obs = True
                    
        if _locked_obs is None: _locked_obs = current_obs
        elif current_obs: _locked_obs = True
        obs = _locked_obs

        final_decision = "Waiting..."
        if light_status in ["Stop (Red)", "Yellow"]:
            final_decision = "Wait for Left -> TURN LEFT" if not obs else "Wait for Straight -> GO STRAIGHT"
        elif light_status == "Left (Red+Green)":
            final_decision = "ACTION: TURN LEFT NOW" if not obs else "Wait for Straight -> GO STRAIGHT"
        elif light_status == "Straight (Green)":
            final_decision = "Wait for Left -> TURN LEFT" if not obs else "ACTION: GO STRAIGHT NOW"

        if "ACTION: TURN LEFT NOW" in final_decision:
            _is_stopped_at_line = False
            _ignore_stopline_counter = 50
            _mission_status = "left"
            print(f"[좌회전] 신호! (장애물:{obs}) -> Shortcut 진입")
        elif "ACTION: GO STRAIGHT NOW" in final_decision:
            _is_stopped_at_line = False
            _ignore_stopline_counter = 50
            _mission_status = "drive"
            print(f"[직진] 신호! (장애물:{obs}) -> 직진 주행")

        cv2.rectangle(display_img, (roi_tl_x1, roi_tl_y1), (roi_tl_x2, roi_tl_y2), (255, 255, 255), 2)
        cv2.putText(display_img, f"Light: {light_status}", (roi_tl_x1 + 10, roi_tl_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display_img, f"Obs: {obs} | Dec: {final_decision}", (roi_tl_x1 + 10, roi_tl_y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    # ----------------------------------------------------------
    # [B] 보행자 인식 시 정지
    # ----------------------------------------------------------
    elif pedestrian_stop:
        if not was_pedestrian_stop:
            saved_speed = prev_normal_speed
            saved_angle = prev_normal_angle
            was_pedestrian_stop = True
        
        calc_speed = 0.0
        calc_angle = 0.0
        status_text = f"PERSON DETECTED ({pedestrian_count} pts)"
        text_color = (0, 165, 255)
        
        cv2.putText(display_img, status_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
    # ----------------------------------------------------------
    # [C] 주행 상태 (정상, 추월)
    # ----------------------------------------------------------
    else:
        if was_pedestrian_stop:
            calc_speed = saved_speed
            calc_angle = saved_angle
            was_pedestrian_stop = False
            status_text = "RESUMING FROM STOP"
            
        elif overtake_state == 1:
            calc_speed = 10.0 
            overtake_kp = 1.2
            calc_angle = float(error * overtake_kp)
                
            if overtake_direction == "RIGHT":
                if passing_left_count == 0 and (current_time - overtake_step_start_time) > 1.5:
                    if passing_clear_time == 0.0: passing_clear_time = current_time
                    elif (current_time - passing_clear_time) > 0.5:
                        overtake_state = 2
                        overtake_step_start_time = current_time
                        passing_clear_time = 0.0
                else: passing_clear_time = 0.0
            elif overtake_direction == "LEFT":
                if passing_right_count == 0 and (current_time - overtake_step_start_time) > 1.5:
                    if passing_clear_time == 0.0: passing_clear_time = current_time
                    elif (current_time - passing_clear_time) > 0.8:
                        overtake_state = 2
                        overtake_step_start_time = current_time
                        passing_clear_time = 0.0
                else: passing_clear_time = 0.0
            status_text = f"OVT: PASSING ({overtake_direction})"
            cv2.putText(display_img, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        elif overtake_state == 2:
            calc_speed = 10.0
            overtake_kp = 0.4  
            calc_angle = float(error * overtake_kp)
            status_text = f"OVT: RETURN ({overtake_direction})"
            cv2.putText(display_img, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            if is_yellow_70 and (current_time - overtake_step_start_time) > 1.5:
                overtake_state = 0
                
        # ==========================================================
        # 정상 주행
        # ==========================================================
        else:
            if overtake_state == 0.5:
                status_text = "OVERTAKE PENDING (WAIT STRAIGHT)"
                cv2.putText(display_img, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if not is_lane_corrupted:
                adaptive_kp = base_kp
                if lane_shift_rate > 10 or abs(error) > int(w * 0.05):
                    adaptive_kp += (lane_shift_rate * 0.02) + (abs(error) * 0.004)
                    adaptive_kp = min(adaptive_kp, 1.2)

                calc_angle = float(error * adaptive_kp)
                
                angle_penalty = abs(calc_angle) * 0.32
                shift_penalty = lane_shift_rate * 0.15
                calc_speed = base_speed - angle_penalty - shift_penalty

                # 급커브(2.0) / 일반 커브(8.0) / 직진 하한선(10.0) 보장
                if abs(calc_angle) > 35.0 or lane_shift_rate > 35:
                    calc_speed = 5.0 # 꺾기에 집중하기 위해 강제 초저속 강하
                elif abs(calc_angle) > 15.0 or lane_shift_rate > 12:
                    calc_speed = min(calc_speed, 8.0) # 일반 커브
                else:
                    calc_speed = max(10.0, calc_speed) # 직진 코스
                    
                calc_speed = max(8.0, calc_speed) # 절대 최저속도
            else:
                # 차선 심하게 훼손 시 속도 2.0로 낮추어 조향 집중
                calc_speed = 2.0 
                left_weight = np.sum(mask_lane[scan_y:, :mid_x] == 255)
                right_weight = np.sum(mask_lane[scan_y:, mid_x:] == 255)

                if right_weight > left_weight * 1.3: calc_angle = -60.0
                elif left_weight > right_weight * 1.3: calc_angle = 60.0
                else:
                    calc_angle = float(error * base_kp)
                    calc_speed = 4.0

        if _is_yellow_zone:
            calc_speed = 5.0
            if _yellow_exit_time > 0.0:
                remain_time = max(0.0, 2.0 - (current_time - _yellow_exit_time))
                status_text += f" {remain_time:.1f}s]"
            else:
                status_text += " [SCHOOL ZONE]"
            text_color = (0, 255, 255)  

        if overtake_state == 0 and not was_pedestrian_stop:
            prev_normal_speed = calc_speed
            prev_normal_angle = calc_angle

    # ==========================================================
    # 6. 디스플레이
    # ==========================================================
    cv2.rectangle(display_img, (roi_stop_x1, roi_stop_y1), (roi_stop_x2, roi_stop_y2), (255, 255, 255), 2)
    if _ignore_stopline_counter == 0:
        cv2.putText(display_img, f"White Ratio: {white_ratio:.1f}%", (roi_stop_x1, roi_stop_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.putText(display_img, f"Cooldown: {_ignore_stopline_counter}", (roi_stop_x1, roi_stop_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.line(display_img, (0, scan_y), (w, scan_y), (0, 255, 0), 2)
    cv2.line(display_img, (0, scan_y_70), (w, scan_y_70), (0, 150, 150), 1)
    
    for c in yellow_pts_70:
        cv2.circle(display_img, (c, scan_y_70), 8, (0, 255, 255), -1)

    if not is_lane_corrupted or overtake_state != 0:
        cv2.circle(display_img, (final_L, scan_y), 6, (255, 0, 0), -1)
        cv2.circle(display_img, (final_M, scan_y), 6, (0, 255, 255), -1)
        cv2.circle(display_img, (final_R, scan_y), 6, (255, 0, 255), -1) 
    
    cv2.circle(display_img, (mid_x, scan_y), 5, (255, 255, 255), -1)
    
    # 디스플레이용 타겟점
    int_target_x = max(min(int(mid_x + (calc_angle / (0.4 if not is_lane_corrupted else 1.0))), w), 0)
    cv2.circle(display_img, (int_target_x, scan_y), 6, (0, 0, 255), -1) 
    cv2.line(display_img, (mid_x, scan_y), (int_target_x, scan_y), (0, 0, 255), 2)

    cv2.putText(display_img, f"Status: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(display_img, f"Speed: {calc_speed:.1f} | Angle: {calc_angle:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Driving Monitor", display_img)
    cv2.waitKey(1)

    return calc_angle, calc_speed, _mission_status, final_L, final_M, final_R