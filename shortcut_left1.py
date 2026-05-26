# -*- coding: utf-8 -*-
def shortcut_left1_mission():
    """ 교차로 중심점을 축으로 삼아 90도 내측으로 정밀 진입 턴을 수행 """
    # TODO: 각도 보정 또는 타이머 기동 제어부 설계 
    is_turn_completed = False
    
    if is_turn_completed:
        return 0.0, 15.0, "passed"
    return -30.0, 15.0, "turning" 