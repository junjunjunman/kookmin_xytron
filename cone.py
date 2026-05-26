# -*- coding: utf-8 -*-
import numpy as np

def cone_mission(lidar_data):
    """ 전방 라이다 스캔 데이터의 좌우 거리를 계산하여 가상의 중심선 벽 추종 조향각 생성 """
    # TODO: 양측 라바콘 클러스터 중심 포인트 추종 알고리즘 설계 
    is_passed_cones = False 
    
    if is_passed_cones:
        return 0.0, 15.0, "passed" 
    return 0.0, 15.0, "driving" 