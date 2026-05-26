#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

class ConeDriverNode(Node):
    def __init__(self):
        super().__init__('cone_driver')
        self.get_logger().info('🚀 라바콘 중심 주행 시작!')
        
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        
        self.lidar_ranges = None
        self.motor_msg = XycarMotor()
        self.timer = self.create_timer(0.05, self.control_loop)

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges

    def drive(self, angle, speed):
        self.motor_msg.angle = float(max(-100.0, min(100.0, float(angle))))
        self.motor_msg.speed = float(max(-50.0, min(50.0, float(speed))))
        self.motor_pub.publish(self.motor_msg)

    def control_loop(self):
        if self.lidar_ranges is None:
            return

        ranges = np.array(self.lidar_ranges, dtype=np.float32)

        # 1. 필터링
        ranges[np.isinf(ranges)] = 0.0
        ranges[ranges < 0.35]    = 0.0
        ranges[ranges > 3.0]     = 0.0

        # 2. 섹터 분할
        front_idx = list(range(0, 30)) + list(range(330, 360))
        right_idx = list(range(30, 150))
        left_idx  = list(range(210, 330))

        front_ranges = ranges[front_idx]
        right_ranges = ranges[right_idx]
        left_ranges  = ranges[left_idx]

        def valid_mean(arr):
            valid = arr[arr > 0.0]
            return float(np.mean(valid)) if len(valid) > 0 else 5.0

        def valid_min(arr):
            valid = arr[arr > 0.0]
            return float(np.min(valid)) if len(valid) > 0 else 5.0

        mean_left  = valid_mean(left_ranges)
        mean_right = valid_mean(right_ranges)
        min_left   = valid_min(left_ranges)
        min_right  = valid_min(right_ranges)
        min_front  = valid_min(front_ranges)

        left_count  = int(np.sum(left_ranges > 0.0))
        right_count = int(np.sum(right_ranges > 0.0))

        # 3. ✅ 거리 기반 동적 게인
        #    가까운 쪽 거리가 작을수록 게인이 커짐
        closest = min(min_left, min_right)
        if closest < 0.5:
            gain = 200.0   # 매우 가까움 → 강하게 꺾기
        elif closest < 0.8:
            gain = 150.0   # 가까움
        elif closest < 1.2:
            gain = 100.0    # 보통
        else:
            gain = 70.0    # 멀면 부드럽게

        error = float(mean_right - mean_left)
        angle = error * gain
        angle = float(max(-100.0, min(100.0, angle)))
        speed = 5.0

        # 4. ✅ 너무 가까우면 속도도 줄이기
        if closest < 0.5:
            speed = 3.0
        elif closest < 0.7:
            speed = 4.0

        # 5. 정면 장애물 감속
        if min_front < 0.6:
            self.get_logger().warn(f"🚧 정면 장애물! {min_front:.2f}m")
            speed = 2.0

        # 6. 한쪽 미감지 시 회피
        if left_count == 0 and right_count > 0:
            angle = 60.0
            self.get_logger().warn("👁️ 왼쪽 라바콘 미감지 → 왼쪽 조향")
        elif right_count == 0 and left_count > 0:
            angle = -60.0
            self.get_logger().warn("👁️ 오른쪽 라바콘 미감지 → 오른쪽 조향")

        self.get_logger().info(
            f"좌평균: {mean_left:.2f}({left_count}개) | "
            f"우평균: {mean_right:.2f}({right_count}개) | "
            f"gain: {gain:.0f} | angle: {angle:.1f} | speed: {speed:.1f}"
        )

        self.drive(angle, speed)


def main(args=None):
    rclpy.init(args=args)
    node = ConeDriverNode()
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
