# -*- coding: utf-8 -*-
def shortcut_drive_mission(imu_data):
    """ 차선 유실(장애물 가림) 시 IMU 쿼터니언 Yaw각 성분을 유지하여 오차 누적 없는 직선 맹목 주행 수행 """
    # TODO: 최초 진입 시점의 IMU Yaw값을 잠금 기동한 뒤 데드레코닝 직진 보정 
    intersection_detected = False 
    
    if intersection_detected:
        return 0.0, 15.0, "intersection_reached" 
    return 0.0, 20.0, "driving" 