#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import time
import cv2
import math
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image, LaserScan, Imu
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

# ==========================================
# State Machine 정의 (구간별 상태)
# ==========================================
STATE_THREE_LIGHT = "THREE_LIGHT"       # 1. 3구 신호등
STATE_CONE = "CONE"                     # 2. 라바콘 구간
STATE_STRAIGHT_1 = "STRAIGHT_1"         # 3. 첫번째 직진 (체크보드 이후)
STATE_FOUR_LIGHT = "FOUR_LIGHT"         # 4. 4구 신호등
STATE_STRAIGHT_2 = "STRAIGHT_2"         # 5. 두번째 직진
STATE_ZIGZAG = "ZIGZAG"                 # 6. 지그재그 (+보행자)
STATE_OVERTAKE = "OVERTAKE"             # 7. 추월 구간
STATE_INTERSECTION = "INTERSECTION"     # 8. 교차로 직진 (로직상 STRAIGHT_2나 OVERTAKE 이후로 통합 가능)
STATE_SCHOOL_ZONE = "SCHOOL_ZONE"       # 9. 어린이 보호구역
STATE_TURN_1 = "TURN_1"                 # 10. 첫번째 좌회전
STATE_STRAIGHT_3 = "STRAIGHT_3"         # 11. 세번째 직진
STATE_TURN_2 = "TURN_2"                 # 12. 두번째 좌회전 (이후 체크보드)

# 지름길 전용 상태 (2, 3바퀴째)
STATE_SHORTCUT_LEFT1 = "SHORTCUT_LEFT1" # 16. 지름길 진입 좌회전
STATE_SHORTCUT_DRIVE = "SHORTCUT_DRIVE" # 17. 지름길 직진
STATE_SHORTCUT_LEFT2 = "SHORTCUT_LEFT2" # 18. 지름길 탈출 좌회전 (이후 SCHOOL_ZONE 연결)

STATE_FINISH = "FINISH"                 # 3바퀴 완주 종료

class TrackDriverNode(Node):
    def __init__(self):
        super().__init__('track_driver')
        
        # ROS2 Publisher & Subscriber
        self.motor_pub = self.create_publisher(XycarMotor, '/xycar_motor', 10)
        self.img_sub = self.create_subscription(Image, '/usb_cam/image_raw/front', self.img_callback, qos_profile_sensor_data)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile_sensor_data)
        
        self.bridge = CvBridge()
        
        # 센서 데이터 저장 변수
        self.cv_image = None
        self.lidar_data = None
        self.imu_data = None
        
        # 주행 상태 관리 변수
        self.current_state = STATE_THREE_LIGHT
        self.lap_count = 0
        self.checkboard_cooldown = 0.0 # 중복 인식 방지용 타이머
        
        # 제어 주기: 10Hz (0.1초마다 control_loop 실행)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Track Driver Node Started! Waiting for sensor data...")

    # ==========================================
    # 콜백 함수 (센서 데이터 업데이트)
    # ==========================================
    def img_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges)

    def imu_callback(self, msg):
        self.imu_data = msg

    # ==========================================
    # 메인 제어 루프 (FSM 기반)
    # ==========================================
    def control_loop(self):
        # 센서 데이터가 모두 들어올 때까지 대기
        if self.cv_image is None or self.lidar_data is None:
            return

        angle, speed = 0.0, 0.0
        
        # 전역적으로 체크보드(랩 카운트) 감지
        self.check_lap_count()

        # 상태 머신 (State Machine)
        if self.current_state == STATE_THREE_LIGHT:
            angle, speed, status = self.three()
            if status == "go":
                self.current_state = STATE_CONE

        elif self.current_state == STATE_CONE:
            angle, speed, status = self.cone()
            if status == "passed":
                self.current_state = STATE_STRAIGHT_1

        elif self.current_state == STATE_STRAIGHT_1:
            angle, speed, status = self.straight()
            if status == "stop_line_detected": # 4구 신호등 앞 정지선 발견 시
                self.current_state = STATE_FOUR_LIGHT

        elif self.current_state == STATE_FOUR_LIGHT:
            angle, speed, decision = self.four()
            if decision == "straight":
                self.current_state = STATE_STRAIGHT_2
            elif decision == "left":
                self.current_state = STATE_SHORTCUT_LEFT1

        elif self.current_state == STATE_STRAIGHT_2:
            angle, speed, status = self.straight()
            if status == "zigzag_start": # 지그재그 진입조건 만족 시
                self.current_state = STATE_ZIGZAG

        elif self.current_state == STATE_ZIGZAG:
            angle, speed, status = self.zigzag()
            if status == "passed":
                self.current_state = STATE_OVERTAKE

        elif self.current_state == STATE_OVERTAKE:
            angle, speed, status = self.fast()
            if status == "passed":
                self.current_state = STATE_SCHOOL_ZONE

        elif self.current_state == STATE_SCHOOL_ZONE:
            angle, speed, status = self.slow()
            if status == "passed":
                self.current_state = STATE_TURN_1

        elif self.current_state == STATE_TURN_1:
            angle, speed, status = self.turn("left")
            if status == "passed":
                self.current_state = STATE_STRAIGHT_3

        elif self.current_state == STATE_STRAIGHT_3:
            angle, speed, status = self.straight()
            if status == "curve_start":
                self.current_state = STATE_TURN_2

        elif self.current_state == STATE_TURN_2:
            angle, speed, status = self.turn("left")
            if status == "passed":
                # 턴을 완료하면 다음 바퀴를 위한 첫 직선 코스(4구 신호등 향하는 길)로 변경됨
                # 단, 실제 Lap 갱신은 checkboard 인식 함수가 처리함
                self.current_state = STATE_STRAIGHT_1

        # 지름길 루트 상태
        elif self.current_state == STATE_SHORTCUT_LEFT1:
            angle, speed, status = self.left1()
            if status == "passed":
                self.current_state = STATE_SHORTCUT_DRIVE

        elif self.current_state == STATE_SHORTCUT_DRIVE:
            angle, speed, status = self.shortcut()
            if status == "intersection_reached":
                self.current_state = STATE_SHORTCUT_LEFT2

        elif self.current_state == STATE_SHORTCUT_LEFT2:
            angle, speed, status = self.left2()
            if status == "passed":
                self.current_state = STATE_SCHOOL_ZONE

        elif self.current_state == STATE_FINISH:
            angle, speed = 0.0, 0.0
            self.get_logger().info("🏁 3 Laps Completed! Driving Finished. 🏁")

        # 모터 제어 명령 퍼블리시
        self.publish_motor(angle, speed)

    # ==========================================
    # 주행 알고리즘 함수들
    # ==========================================

    def check_lap_count(self):
        """
        바닥의 체크보드 무늬를 카메라로 검출하여 바퀴 수(lap_count)를 증가시킴.
        """
        if time.time() - self.checkboard_cooldown < 10.0: # 10초 이내 중복 인식 방지
            return

        # TODO: self.cv_image에서 체크보드 패턴 검출 로직 작성
        is_checkboard_detected = False 
        
        if is_checkboard_detected:
            self.lap_count += 1
            self.get_logger().info(f"Lap Updated: {self.lap_count} / 3")
            self.checkboard_cooldown = time.time()
            if self.lap_count >= 3: # 3바퀴(진입 1번 포함 시 카운트 기준 조정 필요)
                self.current_state = STATE_FINISH

    def three(self):
        """ 1. 3구 신호등 탐지 함수 """
        angle, speed = 0.0, 0.0
        status = "wait"
        
        # TODO: self.cv_image의 특정 ROI에서 초록색 픽셀 비율 분석
        is_green_light = False
        
        if is_green_light:
            status = "go"
        return angle, speed, status

    def cone(self):
        """ 2. 라바콘 통과 함수 """
        angle, speed = 0.0, 15.0 # 천천히 전진
        status = "driving"
        
        # TODO: self.lidar_data를 분석하여 좌/우 라바콘 거리를 측정하고 중심점 추종 로직 작성
        # TODO: 아스팔트(노란점선/흰실선)가 보이기 시작하면 통과로 간주
        is_passed_cones = False 
        
        if is_passed_cones:
            status = "passed"
        return angle, speed, status

    def straight(self):
        """ 3, 5, 11. 직선 도로 직진 함수 (오른쪽 차선 주행) """
        angle, speed = 0.0, 30.0
        status = "driving"
        
        # TODO: 왼쪽 노란색 점선, 오른쪽 흰색 실선 인식 (HoughLines, Sliding Window 등)
        # TODO: 멈춤선(가로 흰색 실선), 지그재그 시작점, 커브 시작점 등을 파악하여 status 리턴
        stop_line_detected = False
        
        if stop_line_detected:
            status = "stop_line_detected"
            speed = 0.0 # 일단 멈춤
            
        return angle, speed, status

    def four(self):
        """ 4, 14. 4구 신호등 탐지 및 분기 판단 함수 """
        angle, speed = 0.0, 0.0 # 기본적으로 정지 상태에서 판단
        decision = "wait"
        
        # TODO: 전방 4구 신호등 색상 검출 (빨, 주, 좌, 초)
        light_color = "red" 
        
        # TODO: 좌측 방향 Lidar 데이터를 확인하여 장애물(경찰차) 여부 판별
        left_obstacle_exist = False 
        
        if self.lap_count == 0: # 1번째 바퀴: 무조건 직진
            if light_color == "green":
                decision = "straight"
        else: # 2, 3번째 바퀴
            if left_obstacle_exist: # 경찰차 있음 -> 무조건 직진
                if light_color == "green":
                    decision = "straight"
            else: # 경찰차 없음 -> 신호에 따름
                if light_color == "left_arrow":
                    decision = "left"
                elif light_color == "green":
                    decision = "straight"
                    
        return angle, speed, decision

    def zigzag(self):
        """ 6. 지그재그 구간 함수 """
        # TODO: 내부에서 self.turn()과 self.person()을 활용하여 로직 구성
        # 노란색 점선을 차량 중심에 맞추는 로직 작성
        status = "driving"
        
        # 사람 발견 시 person() 실행 로직 예시
        # if pedestrian_detected:
        #    return self.person()
        
        is_passed_zigzag = False
        if is_passed_zigzag:
            status = "passed"
            
        return 0.0, 20.0, status

    def turn(self, direction):
        """ 6, 10, 12. 곡선 도로 턴 함수 """
        angle, speed = 0.0, 20.0
        status = "driving"
        
        # TODO: 곡선 차선 인식 및 조향 알고리즘. direction 변수("left" or "right")에 맞춰 로직 차별화
        is_turn_completed = False
        
        if is_turn_completed:
            status = "passed"
        return angle, speed, status

    def person(self):
        """ 6. 사람 피하기 함수 """
        # TODO: Lidar로 전방 사람 인식, 좌/우측 공간 파악하여 회피 조향 및 복귀
        status = "avoiding"
        return 0.0, 10.0, status

    def fast(self):
        """ 7. 차량 추월 함수 """
        angle, speed = 0.0, 40.0 # 평소보다 빠르게
        status = "driving"
        
        # TODO: Lidar로 왼쪽/오른쪽 차량 거리 추적, 왼쪽 차선으로 변경 후 추월, 다시 오른쪽 차선 복귀 로직
        is_overtaken = False
        if is_overtaken:
            status = "passed"
        return angle, speed, status

    def slow(self):
        """ 9. 어린이 보호구역 함수 """
        angle, speed = 0.0, 15.0 # 속도 줄임
        status = "driving"
        
        # TODO: 바닥의 "보호구역 해제" 글자 혹은 특정 마커 인식
        is_end_of_school_zone = False
        if is_end_of_school_zone:
            status = "passed"
        return angle, speed, status

    def left1(self):
        """ 16. 4구 신호등 -> 지름길 좌회전 함수 """
        # TODO: 교차로에서 90도 좌회전하여 지름길 차선에 진입하는 로직
        status = "passed" # 턴 완료 시
        return -30.0, 20.0, status

    def shortcut(self):
        """ 17. 지름길 직진 (선 끊김 구간) 함수 """
        angle, speed = 0.0, 30.0
        status = "driving"
        
        # TODO: 장애물로 차선이 가려진 구간. 차선이 안 보일 경우 IMU의 Yaw값을 유지하며 직진하는 로직
        # TODO: 끝단 교차로 인식 시 status 갱신
        intersection_detected = False
        if intersection_detected:
            status = "intersection_reached"
        return angle, speed, status

    def left2(self):
        """ 18. 지름길 끝 -> 어린이보호구역 좌회전 함수 """
        # TODO: 지름길 탈출을 위한 90도 좌회전 로직
        status = "passed" # 턴 완료 시
        return -30.0, 20.0, status

    # ==========================================
    # 모터 제어 퍼블리시
    # ==========================================
    def publish_motor(self, angle, speed):
        # Angle (-100 ~ 100), Speed (-50 ~ 50) Limit
        angle = max(min(angle, 100.0), -100.0)
        speed = max(min(speed, 50.0), -50.0)
        
        msg = XycarMotor()
        msg.angle = float(angle)
        msg.speed = float(speed)
        self.motor_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrackDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        # 종료 시 차량 정지
        node.publish_motor(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
