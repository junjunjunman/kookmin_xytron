#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

class SchoolZoneNode(Node):
    def __init__(self):
        super().__init__('school_zone_driver')
        self.get_logger().info('🏫 어린이보호구역 감지 시작!')

        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        self.image_sub = self.create_subscription(
            Image, '/usb_cam/image_raw/front',
            self.image_callback, qos_profile_sensor_data)

        self.bridge = CvBridge()
        self.motor_msg = XycarMotor()

        self.in_school_zone = False
        self.normal_speed   = 10.0
        self.slow_speed     = 3.0
        self.current_speed  = self.normal_speed

        self.frame_count  = 0
        self.ocr_interval = 3

        self.timer = self.create_timer(0.05, self.control_loop)

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.ocr_interval != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            return

        result = self.detect_zone(frame)

        if result == 'school':
            if not self.in_school_zone:
                self.in_school_zone = True
                self.current_speed  = self.slow_speed
                self.get_logger().warn('🚸 어린이보호구역 진입! 감속')

        elif result == 'release':
            if self.in_school_zone:
                self.in_school_zone = False
                self.current_speed  = self.normal_speed
                self.get_logger().info('✅ 보호구역 해제! 속도 복귀')

    def detect_zone(self, frame):
        h, w = frame.shape[:2]

        # 아래쪽 절반 ROI
        roi = frame[h//2:h, 0:w]

        # 노란색 마스크 (HSV)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 노란 픽셀 총량으로 구분
        yellow_pixels = cv2.countNonZero(mask)

        if yellow_pixels < 300:
            return None

        if yellow_pixels >= 5000:
            return 'school'
        elif yellow_pixels >= 500:
            return 'release'
        else:
            return None

    def drive(self, angle, speed):
        self.motor_msg.angle = float(max(-100.0, min(100.0, float(angle))))
        self.motor_msg.speed = float(max(-50.0,  min(50.0,  float(speed))))
        self.motor_pub.publish(self.motor_msg)

    def control_loop(self):
        self.drive(angle=0.0, speed=self.current_speed)


def main(args=None):
    rclpy.init(args=args)
    node = SchoolZoneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(angle=0.0, speed=0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
