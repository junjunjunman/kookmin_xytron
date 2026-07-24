#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class CamRecorderNode(Node):
    def __init__(self):
        super().__init__('cam_recorder')
        self.bridge = CvBridge()
        
        # 1. 저장 디렉토리 설정 및 생성
        self.save_dir = 'records/recording_2'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 2. 프레임 카운터 및 조향/속도 변수 초기화
        self.frame_count = 0
        self.current_angle = 0.0
        self.current_speed = 0.0

        # 3. CSV 파일 생성 및 헤더(컬럼명) 작성
        self.csv_path = os.path.join(self.save_dir, 'telemetry.csv')
        self.csv_file = open(self.csv_path, mode='w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['frame_id', 'filename', 'angle', 'speed', 'timestamp'])

        # 4. 카메라 이미지 구독 (/image_raw)
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.img_callback,
            10
        )
        
        # 5. 키보드 제어기(control_keyboard.py)의 xycar_motor 토픽 구독
        self.motor_sub = self.create_subscription(
            Float32MultiArray,
            'xycar_motor',
            self.motor_callback,
            10
        )

        self.get_logger().info(f"녹화 시작. 저장 경로: './{self.save_dir}/'")

    def motor_callback(self, msg):
        """xycar_motor 토픽 수신"""
        if len(msg.data) >= 2:
            self.current_angle = float(msg.data[0])
            self.current_speed = float(msg.data[1])

    def img_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        filename = f"frame_{self.frame_count:04d}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, cv_image)
        timestamp = data.header.stamp.sec + (data.header.stamp.nanosec * 1e-9)
        self.csv_writer.writerow([
            self.frame_count,
            filename,
            self.current_angle,
            self.current_speed,
            timestamp
        ])
        
        self.frame_count += 1
        
        # Viewer
        cv2.imshow("Recording Viewer", cv_image)
        cv2.waitKey(1)

    def destroy_node(self):
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
            self.get_logger().info(f"Frame number: {self.frame_count} / Path: './{self.save_dir}/'")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CamRecorderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("=========== Keyboard Interrupt. 녹화 종료 ===========")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
