#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

class StraightTestNode(Node):
    def __init__(self):
        super().__init__('straight_test_node')
        
        # ROS2 구독 및 발행 설정
        self.img_sub = self.create_subscription(
            Image, 
            '/usb_cam/image_raw/front', 
            self.img_callback, 
            qos_profile_sensor_data
        )
        self.motor_pub = self.create_publisher(XycarMotor, '/xycar_motor', 10)
        
        self.bridge = CvBridge()
        self.get_logger().info("🔍 Straight Line (Stopline vs Checkerboard) Test Node Started!")

    def img_callback(self, msg):
        try:
            # ROS Image -> OpenCV BGR 이미지 변환 (680x480)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Bridge Error: {e}")
            return

        h, w, _ = frame.shape
        display_img = frame.copy()

        # 1. HSV 색상 공간 필터링
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # 2. 직선 도로 차선 추종 알고리즘 (Look-ahead 스캔 라인)
        scan_y = int(h * 0.75)
        mid_x = w // 2
        
        left_area = mask_yellow[scan_y, :mid_x]
        right_area = mask_white[scan_y, mid_x:]

        left_indices = np.where(left_area == 255)[0]
        right_indices = np.where(right_area == 255)[0] + mid_x

        ideal_lane_width = int(w * 0.55)
        left_x = mid_x - (ideal_lane_width // 2)
        right_x = mid_x + (ideal_lane_width // 2)

        if len(left_indices) > 0:
            left_x = int(np.mean(left_indices))
        if len(right_indices) > 0:
            right_x = int(np.mean(right_indices))

        if len(left_indices) > 0 and len(right_indices) > 0:
            target_x = (left_x + right_x) // 2
        elif len(left_indices) > 0:
            target_x = left_x + (ideal_lane_width // 2)
        elif len(right_indices) > 0:
            target_x = right_x - (ideal_lane_width // 2)
        else:
            target_x = mid_x

        # P 제어 기반 조향각 연산
        error = target_x - mid_x
        kp = 0.4
        angle = float(error * kp)
        
        # 기본 기본 속도 세팅
        speed = 50.0
        status_text = "DRIVING"
        box_color = (255, 255, 0) # 평상시 차선 추종 상태 (하늘색/노란색 계열)

        # 3. 전방 하단 고정 ROI 내 정지선 / 체크보드 판별 로직
        roi_x_start = int(w * 0.25)
        roi_x_end = int(w * 0.75)
        roi_y_start = int(h * 0.82)
        roi_y_end = int(h * 0.92)

        stop_line_roi = mask_white[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        white_pixel_count = np.sum(stop_line_roi == 255)

        # 바닥에 일정량 이상의 흰색 형상이 포착된 경우 분석 진행 [cite: 53]
        if white_pixel_count > 3000:
            
            # --- 개선 방식 1: 다중 행(Multi-row) 주파수 변환 횟수 측정 ---
            # ROI 내의 상단(25%), 중단(50%), 하단(75%) 세 군데 행을 모두 조사
            h_roi, w_roi = stop_line_roi.shape
            rows_to_check = [int(h_roi * 0.25), int(h_roi * 0.50), int(h_roi * 0.75)]
            
            max_transitions = 0
            for r in rows_to_check:
                sample_row = stop_line_roi[r, :]
                # 픽셀 값이 바뀌는 경계면 개수 연산 [cite: 64, 65]
                transitions = np.sum(np.diff(sample_row.astype(int)) != 0)
                if transitions > max_transitions:
                    max_transitions = transitions

            # --- 개선 방식 2: 외곽선(Contour) 개수 빌드업 검증 ---
            # ROI 영역 내부에서 분리된 흰색 블록(덩어리) 개수를 카운트
            contours, _ = cv2.findContours(stop_line_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 자잘한 노이즈를 필터링하고 면적이 100픽셀 이상인 유효 블록만 추출
            valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
            contour_count = len(valid_contours)

            # --- 최종 교차 판정 ---
            # 독립된 흰색 덩어리가 3개 이상이거나, 행 스캔 중 한 곳이라도 4회 이상 변환이 일어난 경우
            if contour_count >= 3 or max_transitions >= 4:
                status_text = f"CHECKERBOARD (Blobs:{contour_count}, Trans:{max_transitions})"
                box_color = (255, 0, 255) # 보라색 마킹 후 패스 [cite: 63]
                # speed와 angle은 상단 차선 추종 연산 결과값 유지 [cite: 66]
                
            # 단일 통짜 실선으로 판단될 경우 정지선 처리
            else:
                status_text = "STOP LINE DETECTED"
                box_color = (0, 0, 255) # 정지선은 빨간색 마킹 후 차량 정지 [cite: 55]
                speed = 0.0
                angle = 0.0

        # 4. 실시간 모니터링 디스플레이 그리기
        cv2.line(display_img, (0, scan_y), (w, scan_y), (255, 0, 0), 2)
        cv2.circle(display_img, (left_x, scan_y), 8, (0, 255, 255), -1) 
        cv2.circle(display_img, (right_x, scan_y), 8, (255, 255, 255), -1) 
        cv2.circle(display_img, (mid_x, scan_y), 5, (255, 0, 0), -1)
        cv2.circle(display_img, (target_x, scan_y), 6, (0, 0, 255), -1)
        cv2.line(display_img, (mid_x, scan_y), (target_x, scan_y), (0, 255, 0), 3)

        # 감지 박스 오버레이 시각화
        cv2.rectangle(display_img, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), box_color, 2)

        # 모니터링 수치 데이터 UI 매핑
        cv2.putText(display_img, f"Status: {status_text}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.putText(display_img, f"Error: {error} | Angle: {angle:.1f} | Speed: {speed}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display_img, f"White Pixels: {white_pixel_count}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # GUI 레이아웃 출력
        cv2.imshow("Lane & Stopline Tracking", display_img)
        cv2.imshow("White Mask View", mask_white)
        cv2.waitKey(1)

        # 5. 제어 토픽 퍼블리시
        motor_msg = XycarMotor()
        motor_msg.angle = max(min(angle, 100.0), -100.0)
        motor_msg.speed = max(min(speed, 50.0), -50.0)
        self.motor_pub.publish(motor_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StraightTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt - Stopping Test Node')
    finally:
        # 가동 종료 시 강제 셧다운 안전장치
        stop_msg = XycarMotor()
        stop_msg.angle = 0.0
        stop_msg.speed = 0.0
        node.motor_pub.publish(stop_msg)
        
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
