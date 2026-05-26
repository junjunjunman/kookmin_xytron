# -*- coding: utf-8 -*-
def slow_mission(cv_image):
    """ 바닥의 해제 문자 또는 차선 색상이 백색 실선 단일 체제로 완전 복귀됨을 추출하여 해제 판단 """
    # TODO: 적색 아스팔트 컬러 임계 이탈 감지 또는 텍스트 픽셀 검출 
    is_end_of_zone = False 
    
    if is_end_of_zone:
        return 0.0, 15.0, "passed" 
    return 0.0, 10.0, "driving" # 고정 타겟 감속 속도 전달 