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

# ==========================================
# 추월 기동 및 디버깅을 위한 상태 전역 변수
# ==========================================
_lane_state = "normal"
_state_timer = 0

# 시나리오 순서 강제 변수 및 옐로우 존 플래그
_expected_obstacle = "person"  # "person" -> "car" -> "person" 반복
_was_person_detected = False   # 보행자가 지나갔음을 감지하기 위함
_in_yellow_zone = False        # 노란선 감속 구간 판단 플래그

def drive_mission(frame, prev_L, prev_M, prev_R, lidar_ranges):
    """
    일반 주행, 신호등, 교차로, 정지차량 5단계 추월, 65% 단일 스캔라인 기반 노란선 감지 및 
    장애물 순서 강제 인지 기능이 통합된 모듈
    """
    global _is_stopped_at_line, _ignore_stopline_counter, _mission_status
    global _last_normal_angle, _last_normal_speed, _locked_obs
    global _lane_state, _state_timer
    global _expected_obstacle, _was_person_detected, _in_yellow_zone

    if frame is None:
        return 0.0, 0.0, "drive", prev_L, prev_M, prev_R

    h, w, _ = frame.shape
    display_img = frame.copy()
    
    status_to_return = "drive"

    if _ignore_stopline_counter > 0:
        _ignore_stopline_counter -= 1
    if _state_timer > 0:
        _state_timer -= 1

    # 1. HSV 색상 공간 필터링 및 통합 차선 마스크 생성
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_lane = cv2.bitwise_or(mask_yellow, mask_white)

    # 2. 65% 주행 3차선 교점 인식
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

    if len(centers) >= 3: det_L, det_M, det_R = centers[0], centers[1], centers[2]
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

    if det_L is not None and det_M is not None and det_R is not None: final_L, final_M, final_R = det_L, det_M, det_R
    elif det_L is not None and det_M is not None: final_L, final_M, final_R = det_L, det_M, det_M + half_width
    elif det_M is not None and det_R is not None: final_L, final_M, final_R = det_M - half_width, det_M, det_R
    elif det_L is not None and det_R is not None: final_L, final_M, final_R = det_L, (det_L + det_R) // 2, det_R
    elif det_L is not None: final_L, final_M, final_R = det_L, det_L + half_width, det_L + ideal_lane_width
    elif det_M is not None: final_L, final_M, final_R = det_M - half_width, det_M, det_M + half_width
    elif det_R is not None: final_L, final_M, final_R = det_R - ideal_lane_width, det_R - half_width, det_R
    else: final_L, final_M, final_R = ref_L, ref_M, ref_R

    lane_shift_rate = 0
    if prev_L is not None and prev_M is not None and prev_R is not None:
        lane_shift_rate = max(abs(final_L - prev_L), abs(final_M - prev_M), abs(final_R - prev_R))

    is_lane_corrupted = (final_L >= final_M) or (final_M >= final_R) or (final_R - final_L > ideal_lane_width * 1.5) or (final_R - final_L < ideal_lane_width * 0.5)

    # ==========================================
    # 2-1. 65% 단일 스캔라인 기반 Yellow Zone 감지
    # ==========================================
    check_window = 15
    is_L_yellow_65 = np.any(mask_yellow[scan_y, max(0, final_L - check_window):min(w, final_L + check_window)] == 255)
    is_R_yellow_65 = np.any(mask_yellow[scan_y, max(0, final_R - check_window):min(w, final_R + check_window)] == 255)
    
    # 일반 주행 모드일 때만 노란선 진입/탈출을 판별
    if _lane_state == "normal":
        if is_L_yellow_65 and is_R_yellow_65:
            if not _in_yellow_zone:
                _in_yellow_zone = True
                print("\n⚠️ [Yellow Zone 진입] 양쪽 노란선 감지 -> 감속 주행 모드\n")
        else:
            if _in_yellow_zone:
                _in_yellow_zone = False
                print("\n🏁 [Yellow Zone 탈출] 노란선 구역 해제 -> 일반 주행 복귀!\n")
    else:
        _in_yellow_zone = False

    # ==========================================
    # 3. 전방 라이다 (시나리오 기반 장애물 정밀 탐지)
    # ==========================================
    person_detected = False
    total_car_pts = 0
    total_person_pts = 0
    car_left_cnt = 0
    car_right_cnt = 0

    if lidar_ranges is not None and len(lidar_ranges) >= 360:
        lidar_np = np.array(lidar_ranges)
        
        idx_front_left = list(range(0, 60))
        idx_front_right = list(range(300, 360))
        
        valid_left = lidar_np[idx_front_left]
        valid_right = lidar_np[idx_front_right]
        
        # 6.0m 탐색 (차량 회피용)
        pts_car_left = valid_left[(valid_left > 0.1) & (valid_left < 6.0)]
        pts_car_right = valid_right[(valid_right > 0.1) & (valid_right < 6.0)]
        car_left_cnt = len(pts_car_left)
        car_right_cnt = len(pts_car_right)
        total_car_pts = car_left_cnt + car_right_cnt
        
        # 1.5m 탐색 (보행자 정지용)
        pts_person_left = valid_left[(valid_left > 0.1) & (valid_left < 1.5)]
        pts_person_right = valid_right[(valid_right > 0.1) & (valid_right < 1.5)]
        total_person_pts = len(pts_person_left) + len(pts_person_right)
        
        # ----------------------------------------------------
        # [핵심] 장애물 순서 강제 로직 (Person -> Car -> Person)
        # ----------------------------------------------------
        if _expected_obstacle == "person":
            if total_person_pts > 0:
                person_detected = True
                if not _was_person_detected:
                    print(f"\n🚶 [장애물 감지] 보행자 발견! 정지합니다. (포인트: {total_person_pts})")
                    _was_person_detected = True
            
            elif _was_person_detected and total_person_pts == 0:
                print("🚶 [장애물 통과] 보행자가 지나갔습니다. 다음 타겟을 '차량'으로 전환합니다!\n")
                _was_person_detected = False
                _expected_obstacle = "car"

        elif _expected_obstacle == "car":
            if total_car_pts >= 10:
                if _lane_state == "normal":
                    print(f"\n🚗 [장애물 감지] 전방 차량 인지 확정! (거리 < 5.0m, 포인트: {total_car_pts} [좌:{car_left_cnt} 우:{car_right_cnt}])")
                    if car_left_cnt > car_right_cnt:
                        _lane_state = "evade_right"
                        _state_timer = 15
                        print("   -> ➡️ 오른쪽(2차선)으로 회피 기동 시작!\n")
                    else:
                        _lane_state = "evade_left"
                        _state_timer = 15
                        print("   -> ⬅️ 왼쪽(1차선)으로 회피 기동 시작!\n")


    # 화면 디버깅 패널 출력
    cv2.putText(display_img, f"[DEBUG] Expected Target: {_expected_obstacle.upper()}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(display_img, f"[DEBUG] Pts(5m) Car: {total_car_pts} (L:{car_left_cnt} R:{car_right_cnt})", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_img, f"[DEBUG] Pts(1.5m) Person: {total_person_pts} -> STOP: {person_detected}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_img, f"[DEBUG] FSM State: {_lane_state} (Timer: {_state_timer})", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_img, f"[DEBUG] Yellow Zone: {_in_yellow_zone}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if person_detected:
        cv2.putText(display_img, f"WARNING: PERSON STOP", (mid_x - 120, int(h * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

    # ==========================================
    # 4. 상태(FSM)에 따른 Target X 설정 및 추월 속도 설정 (5단계)
    # ==========================================
    base_speed = 20.0
    base_kp = 0.4
    boost_kp = 1.0
    status_text = f"DRIVING (Status: {_mission_status})"
    box_color = (255, 255, 0)
    target_x = (final_L + final_R) // 2 

    if _lane_state == "normal":
        target_x = (final_L + final_R) // 2
        
    # ------------------ (예1) 우측 회피 (좌측 1차선 막힘) ------------------
    elif _lane_state == "evade_right":
        target_x = (final_M + final_R) // 2  # 2차선 진입
        boost_kp = 1.5
        base_speed = 10.0
        status_text = "EVADE TO LANE 2"
        if _state_timer == 0:
            _lane_state = "pass_right"
            _state_timer = 15
            
    elif _lane_state == "pass_right":
        target_x = (final_M + final_R) // 2  # 2차선 주행 (추월)
        base_speed = 20.0
        status_text = "PASSING LANE 2"
        if _state_timer == 0:
            _lane_state = "return_lane1"
            _state_timer = 15
            
    elif _lane_state == "return_lane1":
        target_x = (final_L + final_M) // 2  # 1차선으로 복귀
        boost_kp = 1.5
        base_speed = 10.0
        status_text = "RETURN TO LANE 1"
        if _state_timer == 0:
            _lane_state = "straight_lane1"
            _state_timer = 15

    elif _lane_state == "straight_lane1":
        target_x = (final_L + final_M) // 2  # 1차선에서 직진
        base_speed = 15.0
        status_text = "STRAIGHT IN LANE 1"
        if _state_timer == 0:
            _lane_state = "return_mid"
            _state_timer = 15

    # ------------------ (예2) 좌측 회피 (우측 2차선 막힘) ------------------
    elif _lane_state == "evade_left":
        target_x = (final_L + final_M) // 2  # 1차선 진입
        boost_kp = 1.5
        base_speed = 10.0
        status_text = "EVADE TO LANE 1"
        if _state_timer == 0:
            _lane_state = "pass_left"
            _state_timer = 15
            
    elif _lane_state == "pass_left":
        target_x = (final_L + final_M) // 2  # 1차선 주행 (추월)
        base_speed = 20.0
        status_text = "PASSING LANE 1"
        if _state_timer == 0:
            _lane_state = "return_lane2"
            _state_timer = 15
            
    elif _lane_state == "return_lane2":
        target_x = (final_M + final_R) // 2  # 2차선으로 복귀
        boost_kp = 1.5
        base_speed = 10.0
        status_text = "RETURN TO LANE 2"
        if _state_timer == 0:
            _lane_state = "straight_lane2"
            _state_timer = 15

    elif _lane_state == "straight_lane2":
        target_x = (final_M + final_R) // 2  # 2차선에서 직진
        base_speed = 15.0
        status_text = "STRAIGHT IN LANE 2"
        if _state_timer == 0:
            _lane_state = "return_mid"
            _state_timer = 15

    # ------------------ 공통: 중앙(M) 본래 주행 복귀 ------------------
    elif _lane_state == "return_mid":
        target_x = (final_L + final_R) // 2  # 중앙 M라인 복귀
        boost_kp = 1.2
        base_speed = 15.0
        status_text = "RETURN TO MID"
        if _state_timer == 0:
            _lane_state = "normal"
            print("🏁 [차선 복귀] 중앙(M)라인 정상 주행 복귀")
            if _expected_obstacle == "car":
                _expected_obstacle = "person"
                print("🔄 [타겟 전환] 다음 장애물을 '보행자'로 설정합니다. (옆 차선 무시됨)\n")

    error = target_x - mid_x

    # 5. 정상 차선 추종 제어 연산 
    if not is_lane_corrupted:
        adaptive_kp = base_kp * boost_kp
        if lane_shift_rate > 10 or abs(error) > int(w * 0.05):
            adaptive_kp += (lane_shift_rate * 0.02) + (abs(error) * 0.004)
            adaptive_kp = min(adaptive_kp, 2.0 if boost_kp > 1.0 else 1.2)

        calc_angle = float(error * adaptive_kp)
        
        # [핵심 추가] 너무 많이 꺾여서 차선 밖으로 나가는 현상 방지
        # 타겟을 옮길 때 순간적으로 커지는 각도를 물리적으로 제한합니다 (-40도 ~ +40도)
        if _lane_state != "normal":
            calc_angle = max(min(calc_angle, 40.0), -40.0)

        angle_penalty = abs(calc_angle) * 0.32 
        shift_penalty = lane_shift_rate * 0.15
        calc_speed = base_speed - angle_penalty - shift_penalty
        
        if abs(calc_angle) > 15.0 or lane_shift_rate > 12:
            calc_speed = min(calc_speed, 8.0)
        calc_speed = max(8.0, calc_speed)
    else:
        calc_speed = 4.0 
        left_weight = np.sum(mask_lane[scan_y:, :mid_x] == 255)
        right_weight = np.sum(mask_lane[scan_y:, mid_x:] == 255)
        
        if right_weight > left_weight * 1.3: calc_angle = -30.0 
        elif left_weight > right_weight * 1.3: calc_angle = 30.0   
        else:
            calc_angle = float(error * base_kp)
            calc_speed = 5.5

    # ----------------------------------------------------
    # [핵심] 추월 기동 및 옐로우 존 최종 속도 덮어쓰기
    # ----------------------------------------------------
    if _lane_state != "normal":
        calc_speed = base_speed  # 추월 중에는 페널티를 받지 않고 15.0속도 고정
        
    # 노란선 구간 진입 상태면 무조건 5.0 속도로 감속 주행
    if _in_yellow_zone:
        calc_speed = 5.0

    # 6. 정지선 탐지
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
            _is_stopped_at_line = True 
            _locked_obs = None
            print('\n🛑 [DRIVE] 정지선 발견! 차량 강제 제동 (Lock 일On) 및 교차로 의사결정 대기\n')
        else:
            cv2.putText(display_img, f"White Ratio: {white_ratio*100:.1f}%", (roi_x_start, roi_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 7. 4구 신호등 및 교차로 장애물 의사결정
    if _is_stopped_at_line:
        calc_speed = 0.0
        calc_angle = 0.0
        status_text = "WAITING FOR SIGNAL..."
        box_color = (0, 0, 255)
        
        roi_x1, roi_x2 = 86, 428
        roi_y1, roi_y2 = 31, 153
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
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

        current_obs = False
        if lidar_ranges is not None and len(lidar_ranges) > 0:
            for i, d in enumerate(lidar_ranges):
                if not math.isfinite(d) or d <= 0.05: continue
                if 20 <= i <= 80 and d < 5.5:
                    current_obs = True
                    break
                    
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
            print(f"\n🟢 [DRIVE] 좌회전 신호! (장애물:{obs}) -> Lock 해제 및 지름길 모드 진입 (Status: left)\n")
            
        elif "ACTION: GO STRAIGHT NOW" in final_decision:
            _is_stopped_at_line = False
            _ignore_stopline_counter = 50
            _mission_status = "drive"
            print(f"\n🟢 [DRIVE] 직진 신호! (장애물:{obs}) -> Lock 해제 및 직진 주행 (Status: drive)\n")

        cv2.rectangle(display_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(display_img, f"Light: {light_status}", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(display_img, f"Obs: {obs} | Dec: {final_decision}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # 8. 최종 조향각/속도 할당
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
            
    status_to_return = _mission_status

    # 9. 디스플레이 렌더링
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

    if _ignore_stopline_counter > 0:
        cv2.rectangle(display_img, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)
        cv2.putText(display_img, f"IGNORE LINE: {_ignore_stopline_counter}", (roi_x_start, roi_y_start-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        cv2.rectangle(display_img, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), box_color, 2)

    cv2.putText(display_img, f"Mode: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    cv2.putText(display_img, f"Angle: {angle:.1f} | Speed: {speed:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Lane & Traffic Light Tracker", display_img)
    cv2.waitKey(1)

    return angle, float(speed), status_to_return, final_L, final_M, final_R
