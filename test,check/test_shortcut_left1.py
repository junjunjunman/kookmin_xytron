#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from xycar_msgs.msg import XycarMotor
import time

class LeftTurnMission(Node):
    def __init__(self):
        super().__init__('left_turn_mission')

        self.cmd_pub = self.create_publisher(XycarMotor, '/xycar_motor', 10)
        self.status_pub = self.create_publisher(String, '/mission_status', 10)

        self.motor_msg = XycarMotor()
        self.get_logger().info("좌회전 미션 시작!")

    def publish_motor(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.cmd_pub.publish(self.motor_msg)

    def stop(self):
        self.publish_motor(angle=0.0, speed=0.0)
        self.get_logger().info("정지!")

    def turn_left(self, speed=5.0, angle=-90.0, duration=5.0):
        self.get_logger().info("좌회전 실행!")
        end_time = time.time() + duration
        while time.time() < end_time:
            self.publish_motor(angle=angle, speed=speed)
            rclpy.spin_once(self, timeout_sec=0.1)  # 이벤트 처리
        self.stop()  # 좌회전 후 정지

    def go_straight(self, speed=10.0, duration=15.0):  # 직진 시간 증가
        self.get_logger().info("직진 중...")
        end_time = time.time() + duration
        while time.time() < end_time:
            self.publish_motor(angle=0.0, speed=speed)
            rclpy.spin_once(self, timeout_sec=0.1)  # 이벤트 처리
        self.stop()  # 직진 후 정지

    def run(self):
        self.get_logger().info("=== 좌회전 주행 미션 시작 ===")

        self.turn_left(speed=5.0, angle=-100.0, duration=5.0)  # 한 번 좌회전
        self.go_straight(speed=10.0, duration=15.0)  # 직진 후 정지

def main():
    rclpy.init()
    mission = LeftTurnMission()
    mission.run()
    mission.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
