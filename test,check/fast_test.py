#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np

# ==================================================================
# ⚡ 주행 속도 상수 재정의 (가이드북 제한 규정 내 최고 속도 마이그레이션)
# ==========================================
SPEED_MAX  = 12.0  # 🚀 추월 시 순간 가속 속도 (최대 12)
SPEED_BASE = 11.0  # 🏎️ 차선 변경 및 패싱 시 기본 속도 (답답하지 않게 10 -> 11 상향)
SPEED_WAIT = 6.0   # 🐢 중앙선 간 보며 대기할 때의 안전 속도

# FSM(유한 상태 기계) 단계 상수 정의
STEP_CENTER_WAIT = 0  
STEP_LANE_CHANGE = 1  
STEP_PASSING     = 2  
STEP_BACK_LANE   = 3  

over_step = STEP_CENTER_WAIT
target_lane = None  
is_logged = False  # 로그가 중복으로 계속 찍히는 것을 방지하기 위한 플래그

def fast_mission(lidar_data):
    """
    라이다 데이터를 기반으로 전방 차량을 감지하고, 
    터미널 알림과 함께 고속으로 추월을 수행하는 알고리즘
    """
    global over_step, target_lane, is_logged
    
    num_samples = len(lidar_data)
    if num_samples == 0:
        return 0.0, float(SPEED_BASE), "searching"
        
    mid_idx = num_samples // 2
    scan_range = int(num_samples * (20.0 / 360.0))  # 전방 좌우 약 20도 범위
    
    # 1. 정면 장애물 거리 측정
    front_dist = np.min(lidar_data[mid_idx - scan_range : mid_idx + scan_range])
    if front_dist == 0.0: front_dist = 999.0

    # 2. 정면 기준 좌/우 차선 영역 개별 거리 측정
    left_dist = np.min(lidar_data[mid_idx : mid_idx + scan_range])
    right_dist = np.min(lidar_data[mid_idx - scan_range : mid_idx])
    if left_dist == 0.0: left_dist = 999.0
    if right_dist == 0.0: right_dist = 999.0

    # ==========================================
    # [STEP 0] 내 차선 전방 차량 감시 및 대기
    # ==========================================
    if over_step == STEP_CHECK_FRONT:
        if front_dist < 0.6: 
            return 0.0, -3.0, "EMERGENCY_BRAKE"
            
        # 전방 1.5m 이내에 방해 차량이 나타나면 추월 시퀀스 발동
        if front_dist < 1.5:
            # 📢 [최초 1회 로그 출력] 차량이 인식되었음을 터미널에 명확히 알림!
            if not is_logged:
                # ROS2 표준 로거 형식 대신 팀원들과 가독성을 맞추기 위해 이모지와 함께 출력
                print("🚘 [FAST] 전방 방해 차량 인식 완료! 동적 추월 가속 시퀀스를 가동합니다.")
                is_logged = True
            
            over_step = STEP_LANE_CHANGE
            
        return 0.0, float(SPEED_WAIT), "lane_following"

    # ==========================================
    # [STEP 1] 라이다 기반 뚫려있는 차선 탐색 및 진입
    # ==========================================
    elif over_step == STEP_LANE_CHANGE:
        if left_dist > right_dist:
            angle = -25.0  # 좌측 차선으로 추월 진입
            print("🔀 [FAST] 좌측 차선(1차선)이 더 멀리 뚫려있음 -> 좌회전 회피")
        else:
            angle = 25.0   # 우측 차선으로 추월 진입
            print("🔀 [FAST] 우측 차선(2차선)이 더 멀리 뚫려있음 -> 우회전 회피")
            
        if front_dist > 2.0:
            over_step = STEP_PASSING
            
        return float(angle), float(SPEED_BASE), "changing_lane"

    # ==========================================
    # [STEP 2] ⚡ 추월 차선 진입 후 시원하게 최대 가속 패싱
    # ==========================================
    elif over_step == STEP_PASSING:
        # 옆 차선 차량(방해 차량)이 내 차 측면(60도 범위)에서 완전히 사라졌는지 검사
        side_idx = int(num_samples * (60.0 / 360.0))
        side_dist = np.min(lidar_data[mid_idx - side_idx : mid_idx + side_idx])
        if side_dist == 0.0: side_dist = 999.0
        
        # 앞차를 제치는 순간이므로 조향은 똑바로(0.0) 잡고, 속도는 가장 빠른 SPEED_MAX(12.0)로 질주합니다!
        if side_dist > 2.0:
            over_step = STEP_BACK_LANE
            
        return 0.0, float(SPEED_MAX), "passing_car"

    # ==========================================
    # [STEP 3] 원래 차선 복귀 및 정렬
    # ==========================================
    elif over_step == STEP_BACK_LANE:
        if left_dist > right_dist:
            angle = 20.0   # 다시 우측 원래 차선으로 복귀
        else:
            angle = -20.0  # 다시 좌측 원래 차선으로 복귀
            
        print("✅ [FAST] 추월 미션 성공! 원래 차선으로 복귀하여 정렬합니다.")
        
        # 미션이 완벽히 끝나면 다음 방해 차를 위해 로그 플래그와 단계를 초기화합니다.
        is_logged = False
        over_step = STEP_CHECK_FRONT
        return float(angle), float(SPEED_BASE), "passed"

    return 0.0, float(SPEED_BASE), "error"
