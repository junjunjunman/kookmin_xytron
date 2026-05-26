# -*- coding: utf-8 -*-
def fast_mission(lidar_data):
    """ 왼쪽 차선 스위칭 가속 -> 우측 사각지대에 추월 대상 차가 지나갔음이 잡히면 오른쪽 원래 차선 복귀 """
    # TODO: 라이다의 우측 및 후방 레이아웃 샘플링 모니터링 
    is_overtake_done = False
    
    if is_overtake_done:
        return 0.0, 15.0, "passed" 
    return -15.0, 25.0, "driving" 