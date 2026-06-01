#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

# ==========================================
# 차선 유지 및 신호등/장애물 제어를 위한 전역 변수
# ==========================================
_is_stopped_at_line = False
_ignore_stopline_counter = 0
_mission_status = "drive"
_last_normal_angle = 0.0
_last_normal_speed = 15.0
_locked_obs = None  # 교차로 장애물 인식 상태 래치용 변수

# [추가] 차선 변경 회피 기동을 위한 전역 변수
_lane_mode = "center"  # "center"(중앙 점선), "left"(1차선), "right"(2차선)
_lane_mode_timer = 0   # 회피 차선 유지 타이머


def drive_mission(frame, prev_L, prev_M, prev_R, lidar_ranges):
    """
    일반 주행(차선 유지) 및 4구 신호등(직진/좌회전) 판독, 
    교차로 장애물 인식, 정방향 장애물(보행자/정지차량) 회피를 수행하는 통합 모듈
    """
    global _is_stopped_at_line, _ignore_stopline_counter, _mission_status
    global _last_normal_angle, _last_normal_speed, _locked_obs
    global _lane_mode, _lane_mode_timer

    if frame is None:
        return 0.0, 0.0, "drive", prev_L, prev_M, prev_R

    h, w, _ = frame.shape
    display_img = frame.copy()
    
    status_to_return = "drive"

    # 0. 쿨다운 및 회피 타이머 감소
    if _ignore_stopline_counter > 0:
        _ignore_stopline_counter -= 1
        
    if _lane_mode_timer > 0:
        _lane_mode_timer -= 1
        if _lane_mode_timer == 0:
            _lane_mode = "center"  # 타이머 종료 시 다시 기존 차선(점선)으로 복귀

    # 1. HSV 색상 공간 필터링 및 통합 차선 마스크 생성
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_lane = cv2.bitwise_or(mask_yellow, mask_white)

    # 2. 3차선 교점 인식 알고리즘
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

    # 이전 차선 정보를 기반으로 기준점 계산
    ref_L = prev_L if prev_L is not None else (mid_x - half_width)
    ref_M = prev_M if prev_M is not None else mid_x
    ref_R = prev_R if prev_R is not None else (mid_x + half_width)

    det_L, det_M, det_R = None, None, None

    # 검출된 중심점 수에 따른 차선 할당
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

    # 최종 차선 좌표 계산
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

    # 차선 변동률 확인
    lane_shift_rate = 0
    if prev_L is not None and prev_M is not None and prev_R is not None:
        shift_L = abs(final_L - prev_L)
        shift_M = abs(final_M - prev_M)
        shift_R = abs(final_R - prev_R)
        lane_shift_rate = max(shift_L, shift_M, shift_R)

    # 차선 무결성 검증
    is_lane_corrupted = False
    if (final_L >= final_M) or (final_M >= final_R) or (final_R - final_L > ideal_lane_width * 1.5) or (final_R - final_L < ideal_lane_width * 0.5):
        is_lane_corrupted = True

    base_speed = 20.0
    base_kp = 0.4
    status_text = f"DRIVING (Status: {_mission_status})"
    box_color = (255, 255, 0)
    
    # ==========================================
    # [핵심 변경] 차선 변경 모드에 따른 Target X 설정
    # ==========================================
    boost_kp = 1.0
    if _lane_mode == "left":
        target_x = (final_L + final_M) // 2  # 1차선 (L과 M 사이)
        boost_kp = 1.8  # 차선 변경 시 확 틀기 위해 Kp 증폭
        cv2.putText(display_img, "AVOIDING: SHIFT LEFT (LANE 1)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    elif _lane_mode == "right":
        target_x = (final_M + final_R) // 2  # 2차선 (M과 R 사이)
        boost_kp = 1.8  # 차선 변경 시 확 틀기 위해 Kp 증폭
        cv2.putText(display_img, "AVOIDING: SHIFT RIGHT (LANE 2)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        target_x = (final_L + final_R) // 2  # 기존 중앙 점선(M) 기준 주행
        
    error = target_x - mid_x

    # 3. 정상 차선 추종 제어 연산
    if not is_lane_corrupted:
        adaptive_kp = base_kp * boost_kp  # 차선 회피 중이면 증폭된 Kp 적용
        if lane_shift_rate > 10 or abs(error) > int(w * 0.05):
            adaptive_kp += (lane_shift_rate * 0.02) + (abs(error) * 0.004)
            # 차선 회피 기동 중일 때는 조향 한계치(Limit)를 높여서 더 급격하게 꺾이도록 허용
            max_kp = 2.0 if _lane_mode != "center" else 1.2
            adaptive_kp = min(adaptive_kp, max_kp)

        calc_angle = float(error * adaptive_kp)
        angle_penalty = abs(calc_angle) * 0.32  
        shift_penalty = lane_shift_rate * 0.15
        calc_speed = base_speed - angle_penalty - shift_penalty
        
        # 각도가 크게 틀어지면 속도를 줄여서 안정적인 차선 변경 유도
        if abs(calc_angle) > 15.0 or lane_shift_rate > 12:
            calc_speed = min(calc_speed, 8.0)
        calc_speed = max(7.0, calc_speed)
        
        check_window = 15
        is_L_yellow = np.any(mask_yellow[scan_y, max(0, final_L - check_window):min(w, final_L + check_window)] == 255)
        is_R_yellow = np.any(mask_yellow[scan_y, max(0, final_R - check_window):min(w, final_R + check_window)] == 255)
        
        if is_L_yellow and is_R_yellow:
            calc_speed = 5.0
            
    else:
        calc_speed = 4.0  
        left_weight = np.sum(mask_lane[scan_y:, :mid_x] == 255)
        right_weight = np.sum(mask_lane[scan_y:, mid_x:] == 255)
        
        if right_weight > left_weight * 1.3: calc_angle = -30.0  
        elif left_weight > right_weight * 1.3: calc_angle = 30.0   
        else:
            calc_angle = float(error * base_kp)
            calc_speed = 5.5

    # ==========================================
    # [핵심 변경] 전방 라이다 (보행자 vs 정지차량 구분)
    # ==========================================
    person_detected = False
    
    if lidar_ranges is not None and len(lidar_ranges) >= 360:
        lidar_np = np.array(lidar_ranges)
        
        # 좌우측 70도 범위 인덱스 (0~70도는 좌측, 289~359도는 우측)
        left_idx = list(range(0, 71))
        right_idx = list(range(289, 360))
        
        # 거리 필터링: 0.1m 이상 2.0m 이내 (차량 감지를 위해 기존 1.5m -> 2.0m 연장)
        valid_left = lidar_np[left_idx]
        valid_left = valid_left[(valid_left > 0.1) & (valid_left < 2.0)]
        
        valid_right = lidar_np[right_idx]
        valid_right = valid_right[(valid_right > 0.1) & (valid_right < 2.0)]
        
        total_pts = len(valid_left) + len(valid_right)
        
        if total_pts > 0:
            # 포인트 개수로 보행자(작음) vs 차량(큼) 구분 
            # (※주의: 시뮬레이터 환경에 따라 이 값을 조절해야 할 수 있습니다)
            car_point_threshold = 30  
            
            if total_pts < car_point_threshold:
                # 1. 보행자로 판단 (기존 정지 로직 - 1.5m 이내일 때만 작동)
                front_dists = np.concatenate((valid_left, valid_right))
                if np.min(front_dists) < 1.5:
                    person_detected = True
                    cv2.putText(display_img, f"WARNING: PERSON STOP ({total_pts} pts)", (mid_x - 140, int(h * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            else:
                # 2. 차량으로 판단 (차선 변경 기동)
                # 어느 쪽 차선에 차량이 있는지 파악 (좌/우 포인트 수 비교)
                if len(valid_left) > len(valid_right):
                    _lane_mode = "right"  # 1차선(좌측)에 차량이 있으므로 2차선(우측)으로 회피
                else:
                    _lane_mode = "left"   # 2차선(우측)에 차량이 있으므로 1차선(좌측)으로 회피
                
                # 차량을 지나치고 나서도 약 1.5초(40프레임) 정도 변경된 차선을 유지하여 안전 확보
                _lane_mode_timer = 40  
                cv2.putText(display_img, f"CAR DETECTED: SHIFT {_lane_mode.upper()} ({total_pts} pts)", (mid_x - 180, int(h * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)

    # 5. 정지선 탐지 (쿨다운이 끝났을 때만 검사)
    roi_x_start, roi_x_end = int(w * 0.25), int(w * 0.75)
    roi_y_start, roi_y_end = int(h * 0.70), int(h * 0.80)
    stop_line_roi = mask_white[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    white_ratio = np.sum(stop_line_roi == 255) / ((roi_x_end - roi_x_start) * (roi_y_end - roi_y_start))

    if not _is_stopped_at_line and _ignore_stopline_counter == 0 and not person_detected:
        is_checkerboard = False
        if white_ratio > 0.15: 
            contours, _ = cv2.findContours(stop_line_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_blobs = [c for c in contours if cv2.contourArea(c) > 50]
            if len(valid_blobs) >= 3:
                is_checkerboard = True

        if is_checkerboard:
            box_color = (255, 0, 255) 
        elif white_ratio >= 0.70:
            _is_stopped_at_line = True  # 정지선 Lock On!
            _locked_obs = None # 신규 정지선 진입 시 라이다 장애물 래치 초기화
            print('🛑 [DRIVE] 정지선 발견! 차량 강제 제동 (Lock On) 및 교차로 의사결정 대기')
        else:
            cv2.putText(display_img, f"White Ratio: {white_ratio*100:.1f}%", (roi_x_start, roi_y_start - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 6. 4구 신호등 및 교차로 장애물 의사결정 로직
    if _is_stopped_at_line:
        calc_speed = 0.0
        calc_angle = 0.0
        status_text = "WAITING FOR SIGNAL..."
        box_color = (0, 0, 255)
        
        # 신호등 ROI
        roi_x1, roi_x2 = 86, 428
        roi_y1, roi_y2 = 31, 153
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 신호등 색상 마스크 생성
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

        # 신호 판독
        if is_red_on and is_green_on: light_status = "Left (Red+Green)"
        elif is_yellow_on: light_status = "Yellow"
        elif is_red_on and not is_green_on: light_status = "Stop (Red)"
        elif is_green_on and not is_red_on: light_status = "Straight (Green)"
        else: light_status = "Unknown"

        # 좌측 장애물 (경찰차 등) 판정: 20~80도 범위, 5.5m 이내
        current_obs = False
        if lidar_ranges is not None and len(lidar_ranges) > 0:
            for i, d in enumerate(lidar_ranges):
                if not math.isfinite(d) or d <= 0.05: continue
                if 20 <= i <= 80 and d < 5.5:
                    current_obs = True
                    break
                    
        # 래치 (한번이라도 포착되면 True 유지)
        if _locked_obs is None: _locked_obs = current_obs
        elif current_obs: _locked_obs = True
        obs = _locked_obs

        # Decision 로직
        final_decision = "Waiting..."
        if light_status in ["Stop (Red)", "Yellow"]:
            final_decision = "Wait for Left -> TURN LEFT" if not obs else "Wait for Straight -> GO STRAIGHT"
        elif light_status == "Left (Red+Green)":
            final_decision = "ACTION: TURN LEFT NOW" if not obs else "Wait for Straight -> GO STRAIGHT"
        elif light_status == "Straight (Green)":
            final_decision = "Wait for Left -> TURN LEFT" if not obs else "ACTION: GO STRAIGHT NOW"

        # 주행 허가 시 정지선 Lock 해제 및 모드 전환
        if "ACTION: TURN LEFT NOW" in final_decision:
            _is_stopped_at_line = False
            _ignore_stopline_counter = 50
            _mission_status = "left"
            print(f"🟢 [DRIVE] 좌회전 신호! (장애물:{obs}) -> Lock 해제 및 지름길 모드 진입 (Status: left)")
            
        elif "ACTION: GO STRAIGHT NOW" in final_decision:
            _is_stopped_at_line = False
            _ignore_stopline_counter = 50
            _mission_status = "drive"
            print(f"🟢 [DRIVE] 직진 신호! (장애물:{obs}) -> Lock 해제 및 직진 주행 (Status: drive)")

        # 디스플레이에 신호등/장애물 상태 렌더링
        cv2.rectangle(display_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(display_img, f"Light: {light_status}", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(display_img, f"Obs: {obs} | Dec: {final_decision}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # 7. 최종 조향각/속도 할당
    if person_detected and not _is_stopped_at_line:
        status_text = "PERSON STOP"
        angle = _last_normal_angle
        speed = 0.0
    else:
        angle = calc_angle
        speed = calc_speed
        if not _is_stopped_at_line:
            _last_normal_angle = angle
            _last_normal_speed = speed
            
    # 주행 상태 플래그 확정하여 반환 준비
    status_to_return = _mission_status

    # 8. 디스플레이 렌더링
    cv2.line(display_img, (0, scan_y), (w, scan_y), (255, 0, 0), 2)
    if not is_lane_corrupted:
        cv2.circle(display_img, (final_L, scan_y), 6, (0, 255, 255), -1) 
        cv2.circle(display_img, (final_M, scan_y), 6, (0, 128, 255), -1) 
        cv2.circle(display_img, (final_R, scan_y), 6, (255, 255, 255), -1) 
    
    cv2.circle(display_img, (mid_x, scan_y), 5, (255, 0, 0), -1)
    safe_angle = angle if not (_is_stopped_at_line or person_detected) else 0.0
    int_target_x = max(min(int(mid_x + (safe_angle / (base_kp if not is_lane_corrupted else 1.0))), w), 0)
    cv2.circle(display_img, (int_target_x, scan_y), 6, (0, 0, 255), -1)
    cv2.line(display_img, (mid_x, scan_y), (int_target_x, scan_y), (0, 255, 0), 3)

    # 하단 정지선 검출 박스 그리기
    if _ignore_stopline_counter > 0:
        cv2.rectangle(display_img, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)
        cv2.putText(display_img, f"IGNORE LINE: {_ignore_stopline_counter}", (roi_x_start, roi_y_start-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        cv2.rectangle(display_img, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), box_color, 2)

    cv2.putText(display_img, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    cv2.putText(display_img, f"Angle: {angle:.1f} | Speed: {speed:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Lane & Traffic Light Tracker", display_img)
    cv2.waitKey(1)

    # track_drive.py 규격에 맞게 최종 반환
    return angle, float(speed), status_to_return, final_L, final_M, final_R
