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

        self.lidar_ranges  = None
        self.motor_msg     = XycarMotor()
        self.loop_timer    = self.create_timer(0.05, self.control_loop)

        self.start_time    = self.get_clock().now()
        self.status        = 'driving'   # 'driving' → 'passed'
        self.passed_count  = 0
        self.PASSED_THRESHOLD = 9

        # 진입 감지용: 좌측 거리가 N프레임 연속 가까워야 진입으로 판단
        self.near_count    = 0
        self.was_detecting = False

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges

    def drive(self, angle, speed):
        self.motor_msg.angle = float(np.clip(angle, -100.0, 100.0))
        self.motor_msg.speed = float(np.clip(speed,  -50.0,  50.0))
        self.motor_pub.publish(self.motor_msg)

    def filter_ranges(self, raw):
        ranges = np.array(raw, dtype=np.float32)
        ranges[np.isinf(ranges)] = 0.0
        ranges[ranges < 0.40]    = 0.0
        ranges[ranges > 3.0]     = 0.0

        # 차체 고정 노이즈 인덱스
        noise_idx = (
            list(range(130, 141)) +
            list(range(175, 186)) +
            list(range(220, 231)) +
            list(range(250, 265))
        )
        for i in noise_idx:
            if i < len(ranges):
                ranges[i] = 0.0

        # 우측 0.80m 이하 노이즈 제거
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

    def control_loop(self):
        if self.lidar_ranges is None:
            return
        if self.status == 'passed':
            self.drive(0.0, 0.0)
            return

        ranges = self.filter_ranges(self.lidar_ranges)

        front_idx = np.array(list(range(0, 30)) + list(range(330, 360)))
        right_idx = np.arange(30, 150)
        left_idx  = np.arange(210, 330)

        mean_left,  min_left,  left_count  = self.sector_stats(ranges, left_idx)
        mean_right, min_right, right_count = self.sector_stats(ranges, right_idx)
        _,          min_front, _           = self.sector_stats(ranges, front_idx)

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # ── 진입 감지: 좌측 or 우측에 물체가 3프레임 연속 보이면 진입 ──
        # count 기반이 아니라 min 거리 기반으로 판단
        side_detected = (min_left < 3.0 and left_count >= 3) or \
                        (min_right < 3.0 and right_count >= 3)

        if side_detected:
            self.near_count += 1
        else:
            self.near_count = max(0, self.near_count - 1)

        if self.near_count >= 3 and not self.was_detecting:
            self.was_detecting = True
            self.get_logger().info(f"📍 라바콘 구간 진입! L:{mean_left:.2f}({left_count}) R:{mean_right:.2f}({right_count})")

        # ── 통과 판정: 양쪽 다 안 보이고 충분한 시간 경과 ──
        both_clear = (left_count <= 1 and right_count <= 1)
        if self.was_detecting and both_clear and elapsed > 5.0:
            self.passed_count += 1
            self.get_logger().info(f"🔍 통과 판정 중... ({self.passed_count}/{self.PASSED_THRESHOLD})")
            if self.passed_count >= self.PASSED_THRESHOLD:
                self.get_logger().info("✅ 라바콘 통과! → 정지!")
                self.status = 'passed'
                self.drive(0.0, 0.0)
                return
        else:
            self.passed_count = 0

        # ── 정면 장애물 → 정지 후 후진 ──
        if min_front < 0.45:
            self.get_logger().warn(f"🚧 정면 {min_front:.2f}m → 후진!")
            self.drive(0.0, -5.0)
            return

        # ── 조향 계산 ──
        if not self.was_detecting:
            # 진입 전: 직진
            angle = 0.0
            speed = 9.0

        else:
            # ★ 진입 후: 좌우 min 거리 기반으로 반응형 조향
            # 가까운 쪽에서 멀어지는 방향으로
            closest = min(min_left, min_right)

            if closest < 0.6:
                gain  = 200.0
                speed = 5.0
            elif closest < 1.0:
                gain  = 130.0
                speed = 7.0
            elif closest < 1.5:
                gain  = 80.0
                speed = 9.0
            else:
                gain  = 50.0
                speed = 9.0

            # mean 기반 error (좌가 가까우면 오른쪽으로)
            error = mean_left - mean_right
            angle = float(np.clip(error * gain, -100.0, 100.0))

            # 한쪽 완전 미감지
            if left_count == 0 and right_count >= 2:
                # 왼쪽 라바콘 안 보임 → 왼쪽으로 붙어있을 수 있음 → 오른쪽 유지
                angle = max(angle, 30.0)
                self.get_logger().warn(f"👁️ 좌 미감지 → 우측 유지 angle:{angle:.1f}")
            elif right_count == 0 and left_count >= 2:
                # 오른쪽 라바콘 안 보임 → 왼쪽 유지
                angle = min(angle, -30.0)
                self.get_logger().warn(f"👁️ 우 미감지 → 좌측 유지 angle:{angle:.1f}")

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
