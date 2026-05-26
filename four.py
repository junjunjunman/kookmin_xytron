# -*- coding: utf-8 -*-
import numpy as np

def four_mission(cv_image, lidar_data, lap_count):
    """ 전방 신호 분석 및 2바퀴 이상 진행 시 좌측 경찰차 유무를 연동한 진로 분기 의사결정 """
    # TODO: 좌측 90도 방향 좁은 영역의 LiDAR 거리 수치를 솎아내어 차량 유무 필터링 
    light_color = "red" 
    left_obstacle_exist = False 
    
    if lap_count == 0: 
        if light_color == "green": return 0.0, 15.0, "greenlight" 
    else: 
        if left_obstacle_exist: 
            if light_color == "green": return 0.0, 15.0, "greenlight" 
        else: 
            if light_color == "left_arrow": 
                return -15.0, 15.0, "left"  # 지름길 좌회전 플래그 트리거 발생 
            elif light_color == "green": 
                return 0.0, 15.0, "greenlight" 
                
    return 0.0, 0.0, "wait" 