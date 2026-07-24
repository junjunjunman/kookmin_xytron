#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')

        # =============================================================
        # [1] 라이다 데이터 인덱스 범위 설정
        # =============================================================
        self.start_idx = 1     # 시작 인덱스
        self.end_idx = 504      # 끝 인덱스 (원하는 범위로 조절)

        # =============================================================
        # [2] 화면 시각화 범위 설정 (단위: cm)
        # =============================================================
        self.view_limit = 100   # 반경 150cm (좌우/상하 -150cm ~ +150cm)
        # =============================================================

        # QoS 설정 (LiDAR 토픽 호환성용 Best Effort)
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # LiDAR 데이터 구독
        self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile
        )

        # 데이터 저장 변수
        self.ranges = None
        self.selected_angles = None

        # Matplotlib 설정 (RViz2 스타일)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.ax.set_xlim(-self.view_limit, self.view_limit)
        self.ax.set_ylim(-self.view_limit, self.view_limit)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)  # 거리 격자
        
        # 중심점(로봇 위치) 표기 (빨간 십자가)
        self.ax.plot(0, 0, 'r+', markersize=12, markeredgewidth=2, label='Robot Center')
        
        # 라이다 포인트 (파란 점)
        self.lidar_points, = self.ax.plot([], [], 'bo', markersize=3, label='LiDAR Scan')
        self.ax.legend(loc='upper right')

        plt.ion()
        plt.show()

        #self.get_logger().info(
        #    f"RViz2-Matched Visualizer Ready | Index: [{self.start_idx}:{self.end_idx}]"
        #)

        # 0.1초 타이머 설정
        self.create_timer(0.1, self.timer_callback)

    def lidar_callback(self, msg):
        """LaserScan 데이터를 수신하고 RViz2 좌표계에 맞춰 각도 추출"""
        raw_ranges = np.array(msg.ranges)
        
        # 1. 인덱스 슬라이싱
        end = min(self.end_idx, len(raw_ranges))
        self.ranges = raw_ranges[self.start_idx:end]

        # 2. msg.angle_min과 angle_increment를 이용한 순수 각도 계산 (ROS 표준 기준)
        indices = np.arange(self.start_idx, end)
        self.selected_angles = msg.angle_min + (indices * msg.angle_increment)

    def timer_callback(self):
        """ROS 2 표준 좌표계(X: 앞쪽, Y: 왼쪽)에 맞춰 Matplotlib에 렌더링"""
        if self.ranges is not None and self.selected_angles is not None:
            # 유효성 검사 (inf, nan 및 노이즈 제거)
            valid_mask = np.isfinite(self.ranges) & (self.ranges > 0)
            valid_ranges = self.ranges[valid_mask]
            valid_angles = self.selected_angles[valid_mask]

            # -------------------------------------------------------------
            # [RViz2 좌표계 변환 규칙]
            # ROS 2 기준: 0도는 로봇 정면(+X 축)입니다.
            # Matplotlib 화면 기준:
            #   - 화면 상단(위쪽)  = 로봇 앞쪽 (+X) -> Matplotlib Y축
            #   - 화면 좌측(왼쪽)  = 로봇 왼쪽 (+Y) -> Matplotlib X축
            # -------------------------------------------------------------
            x_ros = valid_ranges * np.cos(valid_angles) * 100  # 로봇 기준 앞/뒤
            y_ros = valid_ranges * np.sin(valid_angles) * 100  # 로봇 기준 좌/우

            # Matplotlib 화면 축 매핑:
            # 화면 X축(가로) = ROS Y축 (왼쪽이 +, 오른쪽이 -)
            # 화면 Y축(세로) = ROS X축 (앞쪽이 +, 뒤쪽이 -)
            plt_x = -y_ros
            plt_y = x_ros

            # 그래프 데이터 업데이트
            self.lidar_points.set_data(plt_x, plt_y)
            self.fig.canvas.draw_idle()
            plt.pause(0.01)

            # 로그 출력
            if len(valid_ranges) > 0:
                ranges_cm = (valid_ranges * 100).astype(int)
                step = max(1, len(ranges_cm) // 10)
                #self.get_logger().info(f"Sample Distance (cm): {ranges_cm[::step].tolist()}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
