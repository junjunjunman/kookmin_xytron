#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

# ── 상태 추적용 모듈 전역 변수 (State Tracking) ──
_start_time = None
_startup_done = False
_startup_done_time = None

turn_mode = False
turn_start = None
passed_state = False

prev_steering = 0.0
last_turn_time = None

# ── 주행 세팅 상수 설정 ──
startup_duration = 5.0
startup_steering = -100.0
startup_speed = 5.0
post_startup_cooldown = 5.0

turn_duration = 5.0
turn_steering = -100.0
turn_speed = 5.0
turn_cooldown = 4.0

base_speed = 9.0
kp = 0.002
steer_alpha = 0.5
lane_half_width = 160


def reset_shortcut_states():
    """지름길 미션이 완료된 후, 다음 바퀴를 위해 상태를 초기화합니다."""
    global _start_time, _startup_done, _startup_done_time
    global turn_mode, turn_start, passed_state
    global prev_steering, last_turn_time

    _start_time = None
    _startup_done = False
    _startup_done_time = None
    turn_mode = False
    turn_start = None
    passed_state = False
    prev_steering = 0.0
    last_turn_time = None


def _make_yellow_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_y = np.array([15, 80, 80])
    upper_y = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_y, upper_y)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _make_white_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_w = np.array([0, 0, 180])
    upper_w = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_w, upper_w)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def detect_u_shape(mask, frame_w):
    """ㅜ자 표지판(지름길 탈출 신호) 감지 알고리즘"""
    if np.count_nonzero(mask) < 300:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 300:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0:
            continue

        aspect_ratio = cw / ch
        if not (1.5 <= aspect_ratio <= 6.0):
            continue

        cx = x + cw // 2
        if not (frame_w * 0.10 < cx < frame_w * 0.90):
            continue

        roi_mask = mask[y:y + ch, x:x + cw]
        if roi_mask.size == 0:
            continue
        rm_h, rm_w = roi_mask.shape

        top_half = roi_mask[:rm_h // 2, :]
        top_col_sum = np.sum(top_half > 0, axis=0)
        top_coverage = np.count_nonzero(top_col_sum > 0) / rm_w
        top_area = np.count_nonzero(top_half)

        bot_half = roi_mask[rm_h // 2:, :]
        bot_col_sum = np.sum(bot_half > 0, axis=0).astype(np.float32)
        cx_s = rm_w // 3
        cx_e = rm_w * 2 // 3
        center_sum = np.sum(bot_col_sum[cx_s:cx_e])
        bot_left_sum = np.sum(bot_col_sum[:cx_s])
        bot_right_sum = np.sum(bot_col_sum[cx_e:])
        total_bot_sum = np.sum(bot_col_sum)

        if total_bot_sum == 0:
            continue

        center_ratio = center_sum / total_bot_sum
        side_ratio = (bot_left_sum + bot_right_sum) / total_bot_sum

        cond_a = (top_coverage > 0.40 and top_area > 150 and center_ratio > 0.30)
        cond_b = (center_ratio > 0.45 and side_ratio < 0.40 and top_coverage > 0.60)

        print(f'[SHORTCUT U-DETECT] ar={aspect_ratio:.2f} top_cov={top_coverage:.2f} '
              f'center_ratio={center_ratio:.2f} side_ratio={side_ratio:.2f} cond_A={cond_a} cond_B={cond_b}')

        if cond_a and cond_b:
            return True

    return False


def shortcut_mission(frame):
    """지름길 통합 모듈 (진입 좌회전 -> 직진 및 탐색 -> 탈출 좌회전)"""
    global _start_time, _startup_done, _startup_done_time
    global turn_mode, turn_start, passed_state
    global prev_steering, last_turn_time

    now = time.time()

    # 이미 지름길 미션을 끝냈다면 바로 passed 리턴
    if passed_state:
        reset_shortcut_states() # 다음 바퀴 진입을 위해 상태 초기화
        return 0.0, 0.0, "passed"

    if _start_time is None:
        _start_time = now

    # ── 1. 초기 강제 좌회전 (지름길 진입) ──
    if not _startup_done:
        elapsed = now - _start_time
        if elapsed < startup_duration:
            return startup_steering, startup_speed, "driving"
        else:
            _startup_done = True
            _startup_done_time = now
            print('[SHORTCUT] Startup left-turn finished')

    # ── 3. ㅜ자 표지판 탈출 좌회전 수행 ── (순서 상 조건문 2, 3 분리)
    if turn_mode:
        elapsed = now - turn_start
        if elapsed < turn_duration:
            return turn_steering, turn_speed, "driving"
        else:
            turn_mode = False
            passed_state = True
            print('[SHORTCUT] U-shape turn completed -> status = passed')
            reset_shortcut_states()
            return 0.0, 0.0, "passed"

    # ── 2. 흰색 실선 사이 주행 및 ㅜ자 표지판 탐색 (지름길 직진) ──
    if frame is None or frame.size == 0:
        return 0.0, base_speed, "driving"

    h, w = frame.shape[:2]

    # ROI 설정
    y1_lane, y2_lane = int(h * 0.60), int(h * 0.95)
    roi_lane = frame[y1_lane:y2_lane, :]

    y1_sign, y2_sign = int(h * 0.25), int(h * 0.75)
    x1_sign, x2_sign = int(w * 0.10), int(w * 0.90)
    roi_sign = frame[y1_sign:y2_sign, x1_sign:x2_sign]

    mask_sign = _make_yellow_mask(roi_sign)

    since = (now - last_turn_time) if last_turn_time is not None else 9999.0
    since_startup = (now - _startup_done_time) if _startup_done_time is not None else 9999.0

    # ㅜ자 감지 로직
    roi_sign_w = x2_sign - x1_sign
    is_u = detect_u_shape(mask_sign, roi_sign_w)

    if is_u:
        if since_startup <= post_startup_cooldown:
            print(f'[SHORTCUT U-DETECT] Suppressed ({since_startup:.1f}s / {post_startup_cooldown}s)')
        elif since > turn_cooldown:
            print('[SHORTCUT U-TURN] Detected -> initiating left turn immediately')
            turn_mode = True
            turn_start = now
            last_turn_time = now
            return turn_steering, turn_speed, "driving"

    # 흰색 차선 추종 로직
    mask_white = _make_white_mask(roi_lane)
    roi_h, roi_w = mask_white.shape

    if np.count_nonzero(mask_white) == 0:
        return 0.0, base_speed, "driving"

    left_mask = mask_white[:, :roi_w // 2]
    right_mask = mask_white[:, roi_w // 2:]

    left_cols = np.where(left_mask > 0)[1]
    right_cols = np.where(right_mask > 0)[1]

    has_left = len(left_cols) > 30
    has_right = len(right_cols) > 30

    if has_left and has_right:
        left_x = int(np.percentile(left_cols, 85))
        right_x = int(np.percentile(right_cols, 15)) + roi_w // 2
        lane_center = (left_x + right_x) // 2
    elif has_left:
        left_x = int(np.mean(left_cols))
        lane_center = left_x + lane_half_width
    elif has_right:
        right_x = int(np.mean(right_cols)) + roi_w // 2
        lane_center = right_x - lane_half_width
    else:
        return 0.0, base_speed, "driving"

    screen_center = roi_w // 2
    error = float(lane_center - screen_center)
    raw_steer = float(max(min(error * kp * roi_w, 50.0), -50.0))
    steering = (steer_alpha * raw_steer) + (1 - steer_alpha) * prev_steering
    prev_steering = steering

    return steering, base_speed, "driving"