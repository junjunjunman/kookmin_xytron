#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
import cv2
import numpy as np

class BasicDriveNode(Node):
    def __init__(self):
        super().__init__('basic_drive_node')
        
        # ROS2 통신 설정: 카메라 토픽 구독 및 모터 토픽 발행 [cite: 182]
        self.subscription = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(XycarMotor, '/xycar_motor', 10)
        self.bridge = CvBridge()
        
        # test_front_view.py의 HSV 임계값 적용 
        self.lower_yellow = np.array([29, 50, 250])
        self.upper_yellow = np.array([31, 255, 255])
        self.lower_white = np.array([0, 0, 250])
        self.upper_white = np.array([0, 20, 255])

        # 이전 차선 교점 위치 저장을 위한 변수
        self.prev_L = None
        self.prev_M = None
        self.prev_R = None

        self.get_logger().info("순수 주행 모드(Basic Drive) 및 뷰어 노드가 시작되었습니다.")

    def image_callback(self, msg):
        try:
            # 1. 이미지 변환 및 마스킹 처리
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w, _ = cv_image.shape
            
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
            mask_lane = cv2.bitwise_or(mask_yellow, mask_white)
            
            # 마스킹된 화면을 배경으로 뷰어 생성
            display_img = cv2.bitwise_and(cv_image, cv_image, mask=mask_lane)
            
            # 2. 3차선 교점 인식 알고리즘 세팅
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

            # 유효한 중심점만 필터링
            centers = [c for c in centers if int(w * 0.08) < c < int(w * 0.92)]
            ideal_lane_width = int(w * 0.48)
            half_width = ideal_lane_width // 2

            # 이전 차선 정보를 기반으로 기준점 계산
            ref_L = self.prev_L if self.prev_L is not None else (mid_x - half_width)
            ref_M = self.prev_M if self.prev_M is not None else mid_x
            ref_R = self.prev_R if self.prev_R is not None else (mid_x + half_width)

            det_L, det_M, det_R = None, None, None

            # 3. 검출된 중심점 수에 따른 차선 할당 (Cost 비교 알고리즘)
            if len(centers) >= 3:
                det_L, det_M, det_R = centers[0], centers[1], centers[2]
            elif len(centers) == 2:
                cost_LM = abs(centers[0] - ref_L) + abs(centers[1] - ref_M)
                cost_MR = abs(centers[0] - ref_M) + abs(centers[1] - ref_R)
                cost_LR = abs(centers[0] - ref_L) + abs(centers[1] - ref_R)
                min_cost = min(cost_LM, cost_MR, cost_LR)
                if min_cost == cost_LM: det_L, det_M = centers[0], centers[1]
                elif min_cost == cost_MR: det_M, det_R = centers[0], centers[1]
                else: det_L, det_R = centers[0], centers[1]
            elif len(centers) == 1:
                cost_L, cost_M, cost_R = abs(centers[0] - ref_L), abs(centers[0] - ref_M), abs(centers[0] - ref_R)
                min_cost = min(cost_L, cost_M, cost_R)
                if min_cost == cost_L: det_L = centers[0]
                elif min_cost == cost_M: det_M = centers[0]
                else: det_R = centers[0]

            # 최종 3차선 픽셀 좌표 도출
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

            # 4. 차선 무결성 검증 및 변동률 확인
            lane_shift_rate = 0
            if self.prev_L is not None:
                lane_shift_rate = max(abs(final_L - self.prev_L), abs(final_M - self.prev_M), abs(final_R - self.prev_R))

            is_lane_corrupted = False
            if (final_L >= final_M) or (final_M >= final_R) or (final_R - final_L > ideal_lane_width * 1.5) or (final_R - final_L < ideal_lane_width * 0.5):
                is_lane_corrupted = True

            # 5. 조향각(Angle) 및 속도(Speed) 계산
            base_speed = 20.0
            base_kp = 0.4
            target_x = (final_L + final_R) // 2
            error = target_x - mid_x
            status_text = "NORMAL DRIVING"
            text_color = (0, 255, 0)

            if not is_lane_corrupted:
                adaptive_kp = base_kp
                if lane_shift_rate > 10 or abs(error) > int(w * 0.05):
                    adaptive_kp += (lane_shift_rate * 0.02) + (abs(error) * 0.004)
                    adaptive_kp = min(adaptive_kp, 1.2)

                calc_angle = float(error * adaptive_kp)
                angle_penalty = abs(calc_angle) * 0.32  
                shift_penalty = lane_shift_rate * 0.15
                calc_speed = base_speed - angle_penalty - shift_penalty
                
                if abs(calc_angle) > 15.0 or lane_shift_rate > 12:
                    calc_speed = min(calc_speed, 8.0)
                calc_speed = max(10.0, calc_speed)
                
            else:
                # 차선이 심하게 꼬였을 경우의 예외 처리
                status_text = "LANE CORRUPTED"
                text_color = (0, 0, 255)
                calc_speed = 4.0  
                left_weight = np.sum(mask_lane[scan_y:, :mid_x] == 255)
                right_weight = np.sum(mask_lane[scan_y:, mid_x:] == 255)
                
                if right_weight > left_weight * 1.3: calc_angle = -30.0  
                elif left_weight > right_weight * 1.3: calc_angle = 30.0   
                else:
                    calc_angle = float(error * base_kp)
                    calc_speed = 5.5

            # 모터 토픽 발행
            motor_msg = XycarMotor()
            motor_msg.angle = calc_angle
            motor_msg.speed = float(calc_speed)
            self.publisher.publish(motor_msg)

            # 이전 좌표 갱신
            self.prev_L, self.prev_M, self.prev_R = final_L, final_M, final_R

            # 6. Driving Monitor 디스플레이 렌더링
            cv2.line(display_img, (0, scan_y), (w, scan_y), (0, 255, 0), 2)  # 녹색 스캔 라인

            if not is_lane_corrupted:
                cv2.circle(display_img, (final_L, scan_y), 6, (255, 0, 0), -1)   # 파란색 (L)
                cv2.circle(display_img, (final_M, scan_y), 6, (0, 255, 255), -1) # 노란색 (M)
                cv2.circle(display_img, (final_R, scan_y), 6, (255, 0, 255), -1) # 분홍색 (R)
            
            # 차량 중앙점과 타겟 목표점 연결
            cv2.circle(display_img, (mid_x, scan_y), 5, (255, 255, 255), -1)
            int_target_x = max(min(int(mid_x + (calc_angle / (base_kp if not is_lane_corrupted else 1.0))), w), 0)
            cv2.circle(display_img, (int_target_x, scan_y), 6, (0, 0, 255), -1) # 빨간색 (Target)
            cv2.line(display_img, (mid_x, scan_y), (int_target_x, scan_y), (0, 0, 255), 2)

            # 텍스트 출력
            cv2.putText(display_img, f"Status: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display_img, f"Speed: {calc_speed:.1f} | Angle: {calc_angle:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Masked Driving Monitor", display_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"이미지 처리 중 오류 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BasicDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 노드가 종료되었습니다.")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
