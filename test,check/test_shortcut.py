#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
import numpy as np
import cv2


def imgmsg_to_cv2_safe(data: Image):
    arr = np.frombuffer(data.data, dtype=np.uint8)
    if data.encoding in ('rgb8', 'bgr8') and data.step >= data.width * 3:
        row = arr.reshape((data.height, data.step))
        frame = row[:, :data.width * 3].reshape((data.height, data.width, 3))
        if data.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    if data.encoding == 'mono8' and data.step >= data.width:
        row = arr.reshape((data.height, data.step))
        frame = row[:, :data.width]
        return frame
    try:
        frame = arr.reshape((data.height, data.step))
        if data.step >= data.width * 3:
            frame = frame[:, :data.width * 3].reshape(
                (data.height, data.width, 3))
        else:
            frame = frame[:, :data.width]
        return frame
    except Exception:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower_node')
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)

        image_topic = '/usb_cam/image_raw/front'
        self.create_subscription(Image, image_topic, self.image_callback, 10)

        self.motor_msg  = XycarMotor()
        self.offset     = 100
        self.base_speed = 9.0
        self.kp         = 0.002

        # ── 첫 번째 강제 좌회전 ───────────────────────────────────
        self.startup_duration   = 5.0
        self.startup_steering   = -100.0
        self.startup_speed      = 5.0
        self._start_time        = None
        self._startup_done      = False
        self._startup_done_time = None

        # ★ 5.0으로 단축: startup 3.5초 + 여유 1.5초
        self.post_startup_cooldown = 5.0

        # ── ㅜ자 표지 좌회전 상태 ────────────────────────────────
        self.turn_mode     = False
        self.turn_start    = None
        self.turn_duration = 5.0
        self.turn_steering = -100.0
        self.turn_speed    = 5.0

        # ── 상태 추적 및 smoothing ────────────────────────────────
        self.status        = ''
        self.passed        = False
        self.prev_steering = 0.0
        self.steer_alpha   = 0.5

        self.last_turn_time  = None
        self.turn_cooldown   = 4.0
        self.lane_half_width = 160

        self.declare_parameter('status', self.status)

    # ── 노란색 마스크 (ㅜ자 표지 감지용) ─────────────────────────
    def _make_yellow_mask(self, roi):
        hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # ★ 흰색 제거 로직 삭제: 반사광이 노란색을 죽이는 부작용 제거
        lower_y = np.array([15, 80, 80])   # ★ 범위 완화 (어두운 환경 대응)
        upper_y = np.array([35, 255, 255])
        mask    = cv2.inRange(hsv, lower_y, upper_y)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    # ── 흰색 마스크 (차선 주행용) ────────────────────────────────
    def _make_white_mask(self, roi):
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_w = np.array([0,   0,  180])
        upper_w = np.array([180, 40, 255])
        mask    = cv2.inRange(hsv, lower_w, upper_w)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    # ── ㅜ자 형태 감지 ───────────────────────────────────────────
    def detect_u_shape(self, mask, frame_w):
        if np.count_nonzero(mask) < 300:
            return False

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 300:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0:
                continue

            aspect_ratio = cw / ch
            if not (1.5 <= aspect_ratio <= 6.0):
                continue

            cx = x + cw // 2
            if not (frame_w * 0.10 < cx < frame_w * 0.90):
                continue

            roi_mask = mask[y:y + ch, x:x + cw]
            if roi_mask.size == 0:
                continue
            rm_h, rm_w = roi_mask.shape

            # 상단 분석
            top_half     = roi_mask[:rm_h // 2, :]
            top_col_sum  = np.sum(top_half > 0, axis=0)
            top_coverage = np.count_nonzero(top_col_sum > 0) / rm_w
            top_area     = np.count_nonzero(top_half)

            # 하단 분석
            bot_half      = roi_mask[rm_h // 2:, :]
            bot_col_sum   = np.sum(bot_half > 0, axis=0).astype(np.float32)
            cx_s          = rm_w // 3
            cx_e          = rm_w * 2 // 3
            center_sum    = np.sum(bot_col_sum[cx_s:cx_e])
            bot_left_sum  = np.sum(bot_col_sum[:cx_s])
            bot_right_sum = np.sum(bot_col_sum[cx_e:])
            total_bot_sum = np.sum(bot_col_sum)

            if total_bot_sum == 0:
                continue

            center_ratio = center_sum / total_bot_sum
            # ★ 핵심: 하단 양쪽 끝이 얼마나 비어있는가
            side_ratio   = (bot_left_sum + bot_right_sum) / total_bot_sum

            # ★ cond_a: 상단 가로막대 존재 확인
            cond_a = (
                top_coverage > 0.40
                and top_area > 150
                and center_ratio > 0.30
            )

            # ★ cond_b: 하단이 중앙에만 집중 (양쪽 끝 비어있음)
            cond_b = (
                center_ratio > 0.45
                and side_ratio < 0.40
                and top_coverage > 0.60
            )

            self.get_logger().info(
                f'[U-DETECT] ar={aspect_ratio:.2f} '
                f'top_cov={top_coverage:.2f} '
                f'center_ratio={center_ratio:.2f} '
                f'side_ratio={side_ratio:.2f} '
                f'cond_A={cond_a} cond_B={cond_b}')

            if cond_a and cond_b:
                return True

        return False

    # ── 메인 콜백 ────────────────────────────────────────────────
    def image_callback(self, data: Image):
        now = self.get_clock().now()
        if self._start_time is None:
            self._start_time = now

        if self.passed:
            self.drive(0.0, 0.0)
            return

        # ── 초기 강제 좌회전 ──────────────────────────────────────
        if not self._startup_done:
            elapsed = (now - self._start_time).nanoseconds / 1e9
            if elapsed < self.startup_duration:
                self.drive(float(self.startup_speed),
                           float(self.startup_steering))
                return
            else:
                self._startup_done      = True
                self._startup_done_time = now
                self.get_logger().info('startup left-turn finished')

        # ── ㅜ자 좌회전 수행 중 ───────────────────────────────────
        if self.turn_mode:
            elapsed = (now - self.turn_start).nanoseconds / 1e9
            if elapsed < self.turn_duration:
                self.drive(float(self.turn_speed), float(self.turn_steering))
                return
            else:
                self.turn_mode = False
                self.status    = 'passed'
                self.passed    = True
                try:
                    self.set_parameters([rclpy.parameter.Parameter(
                        'status',
                        rclpy.Parameter.Type.STRING,
                        self.status)])
                except Exception:
                    pass
                self.drive(0.0, 0.0)
                self.get_logger().info(
                    'u-shape turn completed -> status = passed (stopped)')
                return

        frame = imgmsg_to_cv2_safe(data)
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]

        # ── ROI 설정 ──────────────────────────────────────────────
        y1_lane = int(h * 0.60)
        y2_lane = int(h * 0.95)
        roi_lane = frame[y1_lane:y2_lane, :]

        # ★ ㅜ자 ROI 세로 범위 확장: 0.25~0.75 (표지판 위치 변화 대응)
        y1_sign = int(h * 0.25)
        y2_sign = int(h * 0.75)
        x1_sign = int(w * 0.10)   # ★ 가로도 약간 확장
        x2_sign = int(w * 0.90)
        roi_sign = frame[y1_sign:y2_sign, x1_sign:x2_sign]

        mask_sign = self._make_yellow_mask(roi_sign)

        since = ((now - self.last_turn_time).nanoseconds / 1e9
                 if self.last_turn_time is not None else 9999.0)
        since_startup = (
            (now - self._startup_done_time).nanoseconds / 1e9
            if self._startup_done_time is not None else 9999.0)

        # ── ㅜ자 감지 즉시 좌회전 ────────────────────────────────
        roi_sign_w = x2_sign - x1_sign
        is_u = self.detect_u_shape(mask_sign, roi_sign_w)

        if is_u:
            if since_startup <= self.post_startup_cooldown:
                self.get_logger().info(
                    f'[U-DETECT] suppressed '
                    f'({since_startup:.1f}s / {self.post_startup_cooldown}s)')
            elif since > self.turn_cooldown:
                self.get_logger().info(
                    '[U-TURN] detected -> initiating left turn immediately')
                self.turn_mode      = True
                self.turn_start     = now
                self.last_turn_time = now
                self.drive(float(self.turn_speed), float(self.turn_steering))
                return

        # ── 흰색 실선 두 개 사이 주행 ────────────────────────────
        mask_white = self._make_white_mask(roi_lane)
        roi_h, roi_w = mask_white.shape

        if np.count_nonzero(mask_white) == 0:
            self.get_logger().info('no white lane -> straight')
            self.drive(self.base_speed, 0.0)
            return

        left_mask  = mask_white[:, :roi_w // 2]
        right_mask = mask_white[:, roi_w // 2:]

        left_cols  = np.where(left_mask  > 0)[1]
        right_cols = np.where(right_mask > 0)[1]

        has_left  = len(left_cols)  > 30
        has_right = len(right_cols) > 30

        if has_left and has_right:
            left_x      = int(np.percentile(left_cols,  85))
            right_x     = int(np.percentile(right_cols, 15)) + roi_w // 2
            lane_center = (left_x + right_x) // 2
            self.get_logger().info(
                f'[LANE] both  lx={left_x} rx={right_x} c={lane_center}')

        elif has_left:
            left_x      = int(np.mean(left_cols))
            lane_center = left_x + self.lane_half_width
            self.get_logger().info(
                f'[LANE] left only  lx={left_x} c={lane_center}')

        elif has_right:
            right_x     = int(np.mean(right_cols)) + roi_w // 2
            lane_center = right_x - self.lane_half_width
            self.get_logger().info(
                f'[LANE] right only  rx={right_x} c={lane_center}')

        else:
            self.drive(self.base_speed, 0.0)
            return

        screen_center = roi_w // 2
        error         = float(lane_center - screen_center)
        raw_steer     = float(max(min(error * self.kp * roi_w, 50.0), -50.0))
        steering      = (self.steer_alpha * raw_steer
                         + (1 - self.steer_alpha) * self.prev_steering)
        self.prev_steering = steering
        self.drive(float(self.base_speed), steering)

    def drive(self, speed, steering):
        self.motor_msg.speed = float(speed)
        self.motor_msg.angle = float(steering)
        self.motor_pub.publish(self.motor_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.drive(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
