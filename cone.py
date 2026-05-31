#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np

# ==========================================
# 라바콘 주행 상태 유지를 위한 전역 변수 선언
# ==========================================
_status = 'driving'           # 'driving' -> 'post_pass' -> 'passed'
_passed_count = 0
_PASSED_THRESHOLD = 6

_near_count = 0
_was_detecting = False
_prev_angle = 0.0

_start_time = None
_post_pass_start = None
_POST_PASS_DURATION = 0.8  # 통과 판정 후 추가로 직진할 시간 (초)

# ==========================================
# 라이다 데이터 필터링 및 통계 함수
# ==========================================
def filter_ranges(raw):
    ranges = np.array(raw, dtype=np.float32)
    ranges[np.isinf(ranges)] = 0.0
    ranges[ranges < 0.40]    = 0.0
    ranges[ranges > 3.0]     = 0.0

    # 차체 노이즈 인덱스 배제
    noise_idx = (
        list(range(130, 141)) +
        list(range(175, 186)) +
        list(range(220, 231)) +
        list(range(250, 265))
    )
    for i in noise_idx:
        if i < len(ranges):
            ranges[i] = 0.0

    # 우측 특정 거리 이하 노이즈 제거
    for i in range(30, 150):
        if i < len(ranges) and 0.0 < ranges[i] <= 0.80:
            ranges[i] = 0.0

    return ranges

def sector_stats(ranges, indices):
    arr   = ranges[indices]
    valid = arr[arr > 0.0]
    count = int(len(valid))
    mean  = float(np.mean(valid)) if count > 0 else 5.0
    vmin  = float(np.min(valid))  if count > 0 else 5.0
    return mean, vmin, count

# ==========================================
# 메인 미션 함수
# ==========================================
def cone_mission(lidar_data):
    """
    메인 노드(track_drive.py)에서 10Hz 주기로 호출되는 라바콘 미션 함수
    Returns: (angle, speed, status)
    """
    global _status, _passed_count, _near_count, _was_detecting
    global _prev_angle, _start_time, _post_pass_start

    # 최초 실행 시 타이머 초기화
    if _start_time is None:
        _start_time = time.time()
        print("🚀 [CONE] 라바콘 중심 주행 시작!")

    # 이미 통과 완료 상태라면 계속 통과 신호 반환
    if _status == 'passed':
        return 0.0, 0.0, "passed"

    # ✅ 통과 후 직진 구간 (post_pass) 로직
    if _status == 'post_pass':
        elapsed = time.time() - _post_pass_start
        if elapsed < _POST_PASS_DURATION:
            return 0.0, 9.0, "driving"
        else:
            print("🏁 [CONE] 최종 정지 및 라바콘 미션 종료!")
            _status = 'passed'
            return 0.0, 0.0, "passed"

    # 1. 라이다 데이터 필터링 및 섹터별 분석
    ranges = filter_ranges(lidar_data)

    front_idx = np.array(list(range(0, 30)) + list(range(330, 360)))
    right_idx = np.arange(30, 150)
    left_idx  = np.arange(210, 330)

    mean_left,  min_left,  left_count  = sector_stats(ranges, left_idx)
    mean_right, min_right, right_count = sector_stats(ranges, right_idx)
    _,          min_front, _           = sector_stats(ranges, front_idx)

    elapsed = time.time() - _start_time

    # 2. 진입 감지
    side_detected = (min_left < 3.0 and left_count >= 3) or \
                    (min_right < 3.0 and right_count >= 3)

    if side_detected:
        _near_count += 1
    else:
        _near_count = max(0, _near_count - 1)

    if _near_count >= 3 and not _was_detecting:
        _was_detecting = True
        print(f"📍 [CONE] 라바콘 구간 진입! L:{mean_left:.2f}({left_count}) R:{mean_right:.2f}({right_count})")

    # 3. 통과 판정
    both_clear = (left_count <= 1 and right_count <= 1)
    if _was_detecting and both_clear and elapsed > 5.0:
        _passed_count += 1
        print(f"🔍 [CONE] 통과 판정 중... ({_passed_count}/{_PASSED_THRESHOLD})")
        if _passed_count >= _PASSED_THRESHOLD:
            print("✅ [CONE] 라바콘 통과! → 직진 후 정지 시퀀스 돌입")
            _status = 'post_pass'
            _post_pass_start = time.time()
            return 0.0, 9.0, "driving"
    else:
        _passed_count = 0

    # 4. 정면 장애물 감지 (회피 후진)
    if min_front < 0.65:
        print(f"🚧 [CONE] 정면 장애물 감지 ({min_front:.2f}m) → 후진!")
        _prev_angle = 0.0
        return 0.0, -5.0, "driving"

    # 5. 조향각(Angle) 및 속도(Speed) 계산
    if not _was_detecting:
        # 진입 전 직진
        angle = 0.0
        speed = 9.0
    else:
        error   = min_left - min_right
        closest = min(min_left, min_right)

        # 근접도에 따른 동적 조향 게인(Gain) 설정
        if closest < 0.50:
            gain  = 250.0
            speed = 3.0
        elif closest < 0.65:
            gain  = 180.0
            speed = 4.0
        elif closest < 1.0:
            gain  = 110.0
            speed = 5.0
        elif closest < 1.5:
            gain  = 70.0
            speed = 7.0
        else:
            gain  = 45.0
            speed = 9.0

        raw_angle = float(np.clip(error * gain, -100.0, 100.0))
        MAX_DELTA = 35.0

        # 급조향 방지를 위한 스무딩 처리
        angle = float(np.clip(
            raw_angle,
            _prev_angle - MAX_DELTA,
            _prev_angle + MAX_DELTA
        ))

        # 한쪽 라바콘을 아예 놓쳤을 때의 Fallback 
        if left_count == 0 and right_count >= 2:
            fallback = float(np.clip(min_right * 40.0, 25.0, 60.0))
            angle = max(angle, fallback)
            print(f"👁️ [CONE] 좌 미감지 → 우측 유지 angle:{angle:.1f}")
        elif right_count == 0 and left_count >= 2:
            fallback = float(np.clip(min_left * 40.0, 25.0, 60.0))
            angle = min(angle, -fallback)
            print(f"👁️ [CONE] 우 미감지 → 좌측 유지 angle:{angle:.1f}")

    _prev_angle = angle

    # (선택 사항) 지속적인 디버깅 로그가 필요하면 아래 주석을 해제하세요.
    # print(f"[CONE] angle:{angle:.1f} spd:{speed:.1f} | front:{min_front:.2f}")

    return angle, speed, "driving"