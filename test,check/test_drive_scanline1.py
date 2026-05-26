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
        
        # 차선 추적 연속성 유지 및 소실 대응을 위한 메모리 변수
        self.prev_L = None
        self.prev_M = None
        self.prev_R = None
        
        self.get_logger().info("🚀 Straight Line (Dynamic Curve Deceleration) Test Node Started!")

    def img_callback(self, msg):
        try:
            # ROS Image -> OpenCV BGR 이미지 변환 (680x480)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Bridge Error: {e}")
            return

        h, w, _ = frame.shape
        display_img = frame.copy()

        # 1. HSV 색상 공간 필터링 및 통합 차선 마스크 생성
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_lane = cv2.bitwise_or(mask_yellow, mask_white)

        # 2. 3차선 교점 인식 알고리즘 (Look-ahead 스캔 라인 65% 유지)
        scan_y = int(h * 0.65)
        mid_x = w // 2

        scan_line_row = mask_lane[scan_y, :]
        white_indices = np.where(scan_line_row == 255)[0]

        centers = []
        if len(white_indices) > 0:
            current_cluster = [white_indices[0]]
            for idx in white_indices[1:]:
                if idx - current_cluster[-1] > 20:
                    if len(current_cluster) >= 2:  
                        centers.append(int(np.mean(current_cluster)))
                    current_cluster = [idx]
                else:
                    current_cluster.append(idx)
            if len(current_cluster) >= 2:
                centers.append(int(np.mean(current_cluster)))

        # 아스팔트 도로 바깥 영역 노이즈 필터링 (화면 좌우 8% 마진 배제)
        centers = [c for c in centers if int(w * 0.08) < c < int(w * 0.92)]

        # 화각 65% 높이에 최적화된 차선 간격 가이드라인 설정
        ideal_lane_width = int(w * 0.48)
        half_width = ideal_lane_width // 2

        # 기준 좌표(Reference) 업데이트
        ref_L = self.prev_L if self.prev_L is not None else (mid_x - half_width)
        ref_M = self.prev_M if self.prev_M is not None else mid_x
        ref_R = self.prev_R if self.prev_R is not None else (mid_x + half_width)

        det_L, det_M, det_R = None, None, None

        # 비용 최소화 기반 차선 매칭 및 복구 로직
        if len(centers) >= 3:
            det_L, det_M, det_R = centers[0], centers[1], centers[2]
        elif len(centers) == 2:
            cost_LM = abs(centers[0] - ref_L) + abs(centers[1] - ref_M)
            cost_MR = abs(centers[0] - ref_M) + abs(centers[1] - ref_R)
            cost_LR = abs(centers[0] - ref_L) + abs(centers[1] - ref_R)
            
            min_cost = min(cost_LM, cost_MR, cost_LR)
            if min_cost == cost_LM:
                det_L, det_M = centers[0], centers[1]
            elif min_cost == cost_MR:
                det_M, det_R = centers[0], centers[1]
            else:
                det_L, det_R = centers[0], centers[1]
        elif len(centers) == 1:
            cost_L = abs(centers[0] - ref_L)
            cost_M = abs(centers[0] - ref_M)
            cost_R = abs(centers[0] - ref_R)
            
            min_cost = min(cost_L, cost_M, cost_R)
            if min_cost == cost_L:
                det_L = centers[0]
            elif min_cost == cost_M:
                det_M = centers[0]
            else:
                det_R = centers[0]

        # 기하학적 복구 알고리즘 (Reconstruction)
        if det_L is not None and det_M is not None and det_R is not None:
            final_L, final_M, final_R = det_L, det_M, det_R
        elif det_L is not None and det_M is not None:
            final_L, final_M = det_L, det_M
            final_R = final_M + half_width
        elif det_M is not None and det_R is not None:
            final_M, final_R = det_M, det_R
            final_L = final_M - half_width
        elif det_L is not None and det_R is not None:
            final_L, final_R = det_L, det_R
            final_M = (final_L + final_R) // 2
        elif det_L is not None:
            final_L = det_L
            final_M = final_L + half_width
            final_R = final_L + ideal_lane_width
        elif det_M is not None:
            final_M = det_M
            final_L = final_M - half_width
            final_R = final_M + half_width
        elif det_R is not None:
            final_R = det_R
            final_M = final_R - half_width
            final_L = final_R - ideal_lane_width
        else:
            final_L, final_M, final_R = ref_L, ref_M, ref_R

        # 차선 유동 변동성 감지
        lane_shift_rate = 0
        if self.prev_L is not None and self.prev_M is not None and self.prev_R is not None:
            shift_L = abs(final_L - self.prev_L)
            shift_M = abs(final_M - self.prev_M)
            shift_R = abs(final_R - self.prev_R)
            lane_shift_rate = max(shift_L, shift_M, shift_R)

        # 커브 구간 탈출 시 역조향 방지 전용 기하학 검증
        is_lane_corrupted = False
        if (final_L >= final_M) or (final_M >= final_R) or (final_R - final_L > ideal_lane_width * 1.5) or (final_R - final_L < ideal_lane_width * 0.5):
            is_lane_corrupted = True

        # 기본 주행 세팅
        base_speed = 15.5
        base_kp = 0.4
        status_text = "DRIVING"
        box_color = (255, 255, 0)
        
        target_x = (final_M + final_R) // 2
        error = target_x - mid_x

        # 3. 제어 및 동적 감속 알고리즘 개편
        if not is_lane_corrupted:
            # 동적 조향 상수(Kp) 스케일업
            adaptive_kp = base_kp
            if lane_shift_rate > 10 or abs(error) > int(w * 0.05):
                status_text = "ADAPTIVE CURVE MODE"
                box_color = (0, 165, 255)
                adaptive_kp += (lane_shift_rate * 0.02) + (abs(error) * 0.004)
                adaptive_kp = min(adaptive_kp, 1.2)

            angle = float(error * adaptive_kp)
            
            # [수정 사항 2] 회전각 연동형 능동 감속 로직 (핸들을 꺾는 양에 비례하여 속도 다운)
            angle_penalty = abs(angle) * 0.32   # 조향각이 커질수록 감속 패널티 강화
            shift_penalty = lane_shift_rate * 0.15
            
            speed = base_speed - angle_penalty - shift_penalty
            
            # 급격한 회전이 연속될 경우 최저 속도 가이드라인 강제 타겟팅
            if abs(angle) > 15.0 or lane_shift_rate > 12:
                speed = min(speed, 8.0)
                
            speed = max(5.5, speed) # 최소 안전 속도 보장
            
            # 다음 프레임 정상 데이터 매칭을 위한 전역 변수 업데이트
            self.prev_L = final_L
            self.prev_M = final_M
            self.prev_R = final_R
        else:
            # [수정 사항 1] GUARDRAIL RECOVERY (코너 이탈 예외 프로토콜)
            status_text = "GUARDRAIL RECOVERY"
            box_color = (0, 0, 255)
            
            # 핸들을 확 꺾는 대피 기동이므로 속도를 '확' 줄여 안전을 확보 (4.0 초저속 고정)
            speed = 4.0  
            
            # 스캔라인 하단부 전체의 좌/우 픽셀 밀도(무게 중심) 분석
            left_weight = np.sum(mask_lane[scan_y:, :mid_x] == 255)
            right_weight = np.sum(mask_lane[scan_y:, mid_x:] == 255)
            
            if right_weight > left_weight * 1.3:
                # 우측 벽 감지 -> 좌측으로 핸들을 최대한 확 꺾음
                angle = -30.0  
            elif left_weight > right_weight * 1.3:
                # 좌측 벽 감지 -> 우측으로 핸들을 최대한 확 꺾음
                angle = 30.0   
            else:
                angle = float(error * base_kp)
                speed = 5.5

        # 4. 전방 하단 고정 ROI 내 정지선 / 체크보드 판별 로직
        roi_x_start = int(w * 0.25)
        roi_x_end = int(w * 0.75)
        roi_y_start = int(h * 0.82)
        roi_y_end = int(h * 0.92)

        stop_line_roi = mask_white[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        white_pixel_count = np.sum(stop_line_roi == 255)

        # 커브 인터록 차단 스위치 강화 (조향 제어가 강하게 들어갈 땐 정지선 판별 취소)
        if lane_shift_rate > 10 or abs(angle) > 15.0 or is_lane_corrupted:
            white_pixel_count = 0

        if white_pixel_count > 3000:
            h_roi, w_roi = stop_line_roi.shape
            rows_to_check = [int(h_roi * 0.25), int(h_roi * 0.50), int(h_roi * 0.75)]
            
            max_transitions = 0
            for r in rows_to_check:
                sample_row = stop_line_roi[r, :]
                transitions = np.sum(np.diff(sample_row.astype(int)) != 0)
                if transitions > max_transitions:
                    max_transitions = transitions

            contours, _ = cv2.findContours(stop_line_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for c in contours:
                if cv2.contourArea(c) > 100:
                    x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                    if w_b > h_b * 1.8:
                        valid_contours.append(c)
                        
            contour_count = len(valid_contours)

            if contour_count >= 3 or max_transitions >= 4:
                status_text = f"CHECKERBOARD (Blobs:{contour_count}, Trans:{max_transitions})"
                box_color = (255, 0, 255) 
            elif contour_count > 0:  
                status_text = "STOP LINE DETECTED"
                box_color = (0, 0, 255) 
                speed = 0.0
                angle = 0.0

        # 5. 실시간 모니터링 디스플레이 그리기
        cv2.line(display_img, (0, scan_y), (w, scan_y), (255, 0, 0), 2)
        
        if not is_lane_corrupted:
            cv2.circle(display_img, (final_L, scan_y), 6, (0, 255, 255), -1) 
            cv2.putText(display_img, "L", (final_L - 6, scan_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(display_img, (final_M, scan_y), 6, (0, 128, 255), -1) 
            cv2.putText(display_img, "M", (final_M - 6, scan_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)
            cv2.circle(display_img, (final_R, scan_y), 6, (255, 255, 255), -1) 
            cv2.putText(display_img, "R", (final_R - 6, scan_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.circle(display_img, (mid_x, scan_y), 5, (255, 0, 0), -1)
        
        int_target_x = max(min(int(mid_x + (angle / (base_kp if not is_lane_corrupted else 1.0))), w), 0)
        cv2.circle(display_img, (int_target_x, scan_y), 6, (0, 0, 255), -1)
        cv2.line(display_img, (mid_x, scan_y), (int_target_x, scan_y), (0, 255, 0), 3)

        cv2.rectangle(display_img, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), box_color, 2)

        cv2.putText(display_img, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.putText(display_img, f"Angle: {angle:.1f} | Speed: {speed:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Lane & Stopline Tracking", display_img)
        cv2.imshow("Unified Lane Mask (Yellow+White)", mask_lane)
        cv2.waitKey(1)

        # 6. 제어 토픽 퍼블리시
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
        stop_msg = XycarMotor()
        stop_msg.angle = 0.0
        stop_msg.speed = 0.0
        node.motor_pub.publish(stop_msg)
        
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
