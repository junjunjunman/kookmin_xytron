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
# 개별 파일 분리 모듈 동적 임포트
# ==========================================
try:
    from track_drive.three import three_mission
    from track_drive.cone import cone_mission
    from track_drive.drive import drive_mission
    from track_drive.shortcut import shortcut_mission
except ImportError:
    from three import three_mission
    from cone import cone_mission
    from drive import drive_mission
    from shortcut import shortcut_mission

# ==========================================
# STATE 상수 정의
# ==========================================
STATE_THREE_LIGHT    = "THREE_LIGHT"
STATE_CONE           = "CONE"
STATE_DRIVE          = "DRIVE"
STATE_SHORTCUT       = "SHORTCUT"
STATE_FINISH         = "FINISH"

class TrackDriverNode(Node):
    def __init__(self):
        super().__init__('track_driver') 
        
        self.motor_pub = self.create_publisher(XycarMotor, '/xycar_motor', 10) 
        self.img_sub = self.create_subscription(Image, '/usb_cam/image_raw/front', self.img_callback, qos_profile_sensor_data) 
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data) 
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile_sensor_data) 

        self.bridge = CvBridge() 
        self.cv_image = None 
        self.lidar_data = None 
        self.imu_data = None 
        
        self.current_state = STATE_THREE_LIGHT 
        self.lap_count = 0 
        self.checkboard_cooldown = 0.0 
        
        self.prev_L = None
        self.prev_M = None
        self.prev_R = None
        
        self.shortcut_exit_time = 0.0
        
        self.timer = self.create_timer(0.1, self.control_loop) 
        self.get_logger().info("======= 자율주행 시작! =======") 

    def img_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges) 

    def imu_callback(self, msg):
        self.imu_data = msg 

    def control_loop(self):
        if self.cv_image is None or self.lidar_data is None: 
            return 

        angle, speed = 0.0, 0.0 
        self.check_lap_count()  

        # ------------------------------------------
        # FSM 모듈 호출 처리
        # ------------------------------------------
        if self.current_state == STATE_THREE_LIGHT:
            angle, speed, status = three_mission(self.cv_image)
            if status == "greenlight":
                self.current_state = STATE_CONE

        elif self.current_state == STATE_CONE:
            angle, speed, status = cone_mission(self.lidar_data)
            if status == "passed":
                self.current_state = STATE_DRIVE

        elif self.current_state == STATE_DRIVE:
            angle, speed, status, self.prev_L, self.prev_M, self.prev_R = drive_mission(
                self.cv_image, self.prev_L, self.prev_M, self.prev_R, self.lidar_data
            )
            
            # 지름길 탈출 직후 3초간 speed 5.0
            if time.time() - self.shortcut_exit_time < 3.0:
                speed = 5.0
            
            if status == "left":
                self.current_state = STATE_SHORTCUT

        elif self.current_state == STATE_SHORTCUT:
            angle, speed, status = shortcut_mission(self.cv_image)
            if status == "passed":
                self.current_state = STATE_DRIVE
                
                self.shortcut_exit_time = time.time()
                # shortcut 이후 차선 매핑
                self.prev_L = -140
                self.prev_M = 160
                self.prev_R = 480
                
                self.get_logger().info("[SHORTCUT] 탈출 완료.")

        elif self.current_state == STATE_FINISH:
            angle, speed = 0.0, 0.0 
            self.get_logger().info("🏁 [COMPLETE] 3 Laps Finished Safely!") 

        self.publish_motor(angle, speed) 

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