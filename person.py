# -*- coding: utf-8 -*-
def person_mission(cv_image, lidar_data):
    """ 전방 장애물 우회 기동 수행 후, 후방 또는 측면 라이다 범위를 검사하여 통과 완료 검증 """
    # TODO: 차량의 회피 기동 및 안전 복귀 영역 필터링 락 설계 [cite: 571]
    is_clear = False 
    
    if is_clear:
        return 0.0, 15.0, "passed"
    return 10.0, 10.0, "avoiding" [cite: 571]