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

        self.motor_msg        = XycarMotor()

        self.start_time       = self.get_clock().now()
        self.status           = 'driving'
        self.passed_count     = 0
        self.PASSED_THRESHOLD = 6

        self.near_count       = 0
        self.was_detecting    = False
        self.prev_angle       = 0.0

        self.post_pass_start    = None
        self.POST_PASS_DURATION = 0.8  # ← 이 숫자만 조절해서 위치 맞추기!

    def lidar_callback(self, msg):
        if self.status == 'passed':
            self.drive(0.0, 0.0)
            return

        # ✅ 통과 후 직진 구간
        if self.status == 'post_pass':
            elapsed = (self.get_clock().now() - self.post_pass_start).nanoseconds / 1e9
            if elapsed < self.POST_PASS_DURATION:
                self.drive(0.0, 9.0)
            else:
                self.get_logger().info("🏁 최종 정지!")
                self.drive(0.0, 0.0)
                self.status = 'passed'
            return

        self.control_loop(msg.ranges)

    def drive(self, angle, speed):
        self.motor_msg.angle = float(np.clip(angle, -100.0, 100.0))
        self.motor_msg.speed = float(np.clip(speed,  -50.0,  50.0))
        self.motor_pub.publish(self.motor_msg)

    def filter_ranges(self, raw):
        ranges = np.array(raw, dtype=np.float32)
        ranges[np.isinf(ranges)] = 0.0
        ranges[ranges < 0.40]    = 0.0
        ranges[ranges > 3.0]     = 0.0

        noise_idx = (
            list(range(130, 141)) +
            list(range(175, 186)) +
            list(range(220, 231)) +
            list(range(250, 265))
        )
        for i in noise_idx:
            if i < len(ranges):
                ranges[i] = 0.0

        for i in range(30, 150):
            if i < len(ranges) and 0.0 < ranges[i] <= 0.80:
                ranges[i] = 0.0

        return ranges

    def sector_stats(self, ranges, indices):
        arr   = ranges[indices]
        valid = arr[arr > 0.0]
        count = int(len(valid))
        mean  = float(np.mean(valid)) if count > 0 else 5.0
        vmin  = float(np.min(valid))  if count > 0 else 5.0
        return mean, vmin, count

    def control_loop(self, raw_ranges):
        ranges = self.filter_ranges(raw_ranges)

        front_idx = np.array(list(range(0, 30)) + list(range(330, 360)))
        right_idx = np.arange(30, 150)
        left_idx  = np.arange(210, 330)

        mean_left,  min_left,  left_count  = self.sector_stats(ranges, left_idx)
        mean_right, min_right, right_count = self.sector_stats(ranges, right_idx)
        _,          min_front, _           = self.sector_stats(ranges, front_idx)

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # ── 진입 감지 ──
        side_detected = (min_left < 3.0 and left_count >= 3) or \
                        (min_right < 3.0 and right_count >= 3)

        if side_detected:
            self.near_count += 1
        else:
            self.near_count = max(0, self.near_count - 1)

        if self.near_count >= 3 and not self.was_detecting:
            self.was_detecting = True
            self.get_logger().info(
                f"📍 라바콘 구간 진입! L:{mean_left:.2f}({left_count}) R:{mean_right:.2f}({right_count})")

        # ── 통과 판정 ──
        both_clear = (left_count <= 1 and right_count <= 1)
        if self.was_detecting and both_clear and elapsed > 5.0:
            self.passed_count += 1
            self.get_logger().info(f"🔍 통과 판정 중... ({self.passed_count}/{self.PASSED_THRESHOLD})")
            if self.passed_count >= self.PASSED_THRESHOLD:
                self.get_logger().info("✅ 라바콘 통과! → 직진 후 정지!")
                self.status          = 'post_pass'
                self.post_pass_start = self.get_clock().now()
                self.drive(0.0, 9.0)
                return
        else:
            self.passed_count = 0

        # ── 정면 장애물 감지 ──
        if min_front < 0.65:
            self.get_logger().warn(f"🚧 정면 {min_front:.2f}m → 후진!")
            self.drive(0.0, -5.0)
            self.prev_angle = 0.0
            return

        # ── 조향 계산 ──
        if not self.was_detecting:
            angle = 0.0
            speed = 9.0

        else:
            error   = min_left - min_right
            closest = min(min_left, min_right)

            if closest < 0.50:
                gain  = 250.0
                speed = 3.0
            elif closest < 0.65:
                gain  = 180.0
                speed = 4.0
            elif closest < 1.0:
                gain  = 110.0
                speed = 5.0
            elif closest < 1.5:
                gain  = 70.0
                speed = 7.0
            else:
                gain  = 45.0
                speed = 9.0

            raw_angle = float(np.clip(error * gain, -100.0, 100.0))

            MAX_DELTA = 35.0
            angle = float(np.clip(
                raw_angle,
                self.prev_angle - MAX_DELTA,
                self.prev_angle + MAX_DELTA
            ))

            if left_count == 0 and right_count >= 2:
                fallback = float(np.clip(min_right * 40.0, 25.0, 60.0))
                angle = max(angle, fallback)
                self.get_logger().warn(f"👁️ 좌 미감지 → 우측 유지 angle:{angle:.1f}")
            elif right_count == 0 and left_count >= 2:
                fallback = float(np.clip(min_left * 40.0, 25.0, 60.0))
                angle = min(angle, -fallback)
                self.get_logger().warn(f"👁️ 우 미감지 → 좌측 유지 angle:{angle:.1f}")

        self.prev_angle = angle

        self.get_logger().info(
            f"좌:{mean_left:.2f}({left_count}) 우:{mean_right:.2f}({right_count}) | "
            f"angle:{angle:.1f} spd:{speed:.1f} | "
            f"front:{min_front:.2f} | 감지:{self.was_detecting}({self.near_count}) cnt:{self.passed_count} | {elapsed:.1f}s"
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
        node.drive(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
