#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import time
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image, LaserScan, Imu
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

# ==========================================
# 개별 파일 분리 모듈 동적 임포트 (Import Mission Modules)
# ==========================================
try:
    from track_drive.three import three_mission
    from track_drive.cone import cone_mission
    from track_drive.drive import drive_mission
    from track_drive.four import four_mission
    from track_drive.person import person_mission
    from track_drive.fast import fast_mission
    from track_drive.slow import slow_mission
    from track_drive.shortcut_left1 import shortcut_left1_mission
    from track_drive.shortcut_drive import shortcut_drive_mission
    from track_drive.shortcut_left2 import shortcut_left2_mission
except ImportError:
    # 패키지 실행 환경에 따른 로컬 임포트 예외 처리
    from three import three_mission
    from cone import cone_mission
    from drive import drive_mission
    from four import four_mission
    from person import person_mission
    from fast import fast_mission
    from slow import slow_mission
    from shortcut_left1 import shortcut_left1_mission
    from shortcut_drive import shortcut_drive_mission
    from shortcut_left2 import shortcut_left2_mission

# ==========================================
# 이미지 분석 맵 매핑 전역 상태(STATE) 상수 정의
# ==========================================
STATE_THREE_LIGHT    = "THREE_LIGHT"
STATE_CONE           = "CONE"
STATE_DRIVE          = "DRIVE"
STATE_FOUR_LIGHT     = "FOUR_LIGHT"
STATE_PERSON         = "STATE_PERSON"
STATE_OVERTAKE       = "OVERTAKE"
STATE_SCHOOL_ZONE    = "SCHOOL_ZONE"

STATE_SHORTCUT_LEFT1 = "SHORTCUT_LEFT1"
STATE_SHORTCUT_DRIVE = "SHORTCUT_DRIVE"
STATE_SHORTCUT_LEFT2 = "SHORTCUT_LEFT2"

STATE_FINISH         = "FINISH"

class TrackDriverNode(Node):
    def __init__(self):
        super().__init__('track_driver') 
        
        # ROS2 통신 토픽 구성 
        self.motor_pub = self.create_publisher(XycarMotor, '/xycar_motor', 10) 
        self.img_sub = self.create_subscription(Image, '/usb_cam/image_raw/front', self.img_callback, qos_profile_sensor_data) 
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data) 
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile_sensor_data) 
        
        self.bridge = CvBridge() 
        
        # 센서 원시 데이터 실시간 캐싱 공간 
        self.cv_image = None 
        self.lidar_data = None 
        self.imu_data = None 
        
        # FSM 상태 전역 변수 및 랩 제어 플래그 
        self.current_state = STATE_THREE_LIGHT 
        self.lap_count = 0 
        self.checkboard_cooldown = 0.0 
        
        # drive.py 모듈 연동 차선 연속성 추적 필터 버퍼
        self.prev_L = None
        self.prev_M = None
        self.prev_R = None
        
        # 제어 주기: 10Hz 
        self.timer = self.create_timer(0.1, self.control_loop) 
        self.get_logger().info("Autonomous Driving Node Started!") 

    def img_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges) 

    def imu_callback(self, msg):
        self.imu_data = msg 

    def control_loop(self):
        """ 다이어그램 이벤트 플래그에 맞춰 각 외부 py 모듈 함수를 분기 제어하는 메인 제어 루프 """
        if self.cv_image is None or self.lidar_data is None: 
            return 

        angle, speed = 0.0, 0.0 
        self.check_lap_count()  # 바닥 체크보드 무늬 트래킹

        # ------------------------------------------
        # FSM 모듈 호출 처리 스위치 구문 
        # ------------------------------------------
        if self.current_state == STATE_THREE_LIGHT:
            angle, speed, status = three_mission(self.cv_image)
            if status == "greenlight":
                self.current_state = STATE_CONE

        elif self.current_state == STATE_CONE:
            angle, speed, status = cone_mission(self.lidar_data)
            if status == "passed":
                self.current_state = STATE_DRIVE

        # 일반 주행 함수
        elif self.current_state == STATE_DRIVE:
            angle, speed, status, self.prev_L, self.prev_M, self.prev_R = drive_mission(
                self.cv_image, self.prev_L, self.prev_M, self.prev_R
            )
            
            # 트리거에 따라 STATE 이동
            if status == "stop_line_detected":
                self.current_state = STATE_FOUR_LIGHT
            elif status == "person_detected":
                self.current_state = STATE_PERSON
            elif status == "car_detected":
                self.current_state = STATE_OVERTAKE
            elif status == "schoolzone_detected":
                self.current_state = STATE_SCHOOL_ZONE

        elif self.current_state == STATE_FOUR_LIGHT:
            angle, speed, decision = four_mission(self.cv_image, self.lidar_data, self.lap_count)
            if decision == "greenlight":
                self.current_state = STATE_DRIVE
            elif decision == "left":
                self.current_state = STATE_SHORTCUT_LEFT1

        elif self.current_state == STATE_PERSON:
            angle, speed, status = person_mission(self.cv_image, self.lidar_data)
            if status == "passed":
                self.current_state = STATE_DRIVE

        elif self.current_state == STATE_OVERTAKE:
            angle, speed, status = fast_mission(self.lidar_data)
            if status == "passed":
                self.current_state = STATE_DRIVE

        elif self.current_state == STATE_SCHOOL_ZONE:
            angle, speed, status = slow_mission(self.cv_image)
            if status == "passed":
                self.current_state = STATE_DRIVE

        # 지름길 진입 시
        elif self.current_state == STATE_SHORTCUT_LEFT1:
            angle, speed, status = shortcut_left1_mission()
            if status == "passed":
                self.current_state = STATE_SHORTCUT_DRIVE

        elif self.current_state == STATE_SHORTCUT_DRIVE:
            angle, speed, status = shortcut_drive_mission(self.imu_data)
            if status == "intersection_reached":
                self.current_state = STATE_SHORTCUT_LEFT2

        elif self.current_state == STATE_SHORTCUT_LEFT2:
            angle, speed, status = shortcut_left2_mission()
            if status == "passed":
                self.current_state = STATE_SCHOOL_ZONE

        elif self.current_state == STATE_FINISH:
            angle, speed = 0.0, 0.0 
            self.get_logger().info("🏁 [COMPLETE] 3 Laps Finished Safely!") 

        self.publish_motor(angle, speed) 

    # 메인 루프에서 바닥 체크보드 무늬를 판단해 랩 카운트 세는 함수
    def check_lap_count(self):
        if time.time() - self.checkboard_cooldown < 10.0: 
            return 
        is_checkboard_detected = False
        if is_checkboard_detected:
            self.lap_count += 1 
            self.checkboard_cooldown = time.time() 
            if self.lap_count >= 3: 
                self.current_state = STATE_FINISH 

    def publish_motor(self, angle, speed):
        angle = max(min(float(angle), 100.0), -100.0) 
        speed = max(min(float(speed), 50.0), -50.0) 
        msg = XycarMotor() 
        msg.angle = angle 
        msg.speed = speed 
        self.motor_pub.publish(msg) 

def main(args=None):
    rclpy.init(args=args) 
    node = TrackDriverNode() 
    try:
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_motor(0.0, 0.0) 
        node.destroy_node() 
        rclpy.shutdown() 

if __name__ == '__main__':
    main() 