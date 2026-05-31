#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import XycarMotor
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

class StraightTestNode(Node):
    def __init__(self):
        super().__init__('straight_test_node')
        
        # ROS2 카메라 구독 설정
        self.img_sub = self.create_subscription(
            Image, 
            '/usb_cam/image_raw/front', 
            self.img_callback, 
            qos_profile_sensor_data
        )
        
        # ROS2 라이다 구독 설정
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile_sensor_data
        )
        
        self.motor_pub = self.create_publisher(XycarMotor, '/xycar_motor', 10)
        self.bridge = CvBridge()
        
        # 차선 추적 연속성 유지 변수
        self.prev_L = None
        self.prev_M = None
        self.prev_R = None
        
        # 보행자 정지 시 직전 상태 기억을 위한 변수
        self.lidar_ranges = None
        self.last_normal_angle = 0.0
        self.last_normal_speed = 15.0
        
        # ⏱️ [타임라인 제어 변수]
        # 0: 평상시 기본 차선 주행
        # 1: 판단된 빈 차선으로 10도 탈출 단계 (1.0초)
        # 2: 반대 방향 10도 카운터 복귀 정렬 단계 (1.2초)
        self.evade_stage = 0
        self.stage_start_time = 0.0
        
        # 🏎️ [추월 핵심 변수 세팅]
        self.fixed_evade_angle = 10.0  
        self.evade_direction = "NONE"   # 실시간 동적 방향 저장용 ("LEFT" 또는 "RIGHT")
        
        self.get_logger().info("🚀 Lidar Index Fixed Bi-Directional Overtake Node Started!")

    def lidar_callback(self, msg):
        """ 라이다 데이터를 받아 저장 """
        self.lidar_ranges = np.array(msg.ranges, dtype=np.float32)

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

        # 아스팔트 바깥 영역 노이즈 필터링 (화면 좌우 8% 마진 배제)
        centers = [c for c in centers if int(w * 0.08) < c < int(w * 0.92)]

        ideal_lane_width = int(w * 0.48)
        half_width = ideal_lane_width // 2

        # 기준 좌표(Reference) 업데이트
        ref_L = self.prev_L if self.prev_L is not None else (mid_x - half_width)
        ref_M = self.prev_M if self.prev_M is not None else mid_x
        ref_R = self.prev_R if self.prev_R is not None else (mid_x + half_width)

        det_L, det_M, det_R = None, None, None

        # 비용 최소화 기반 차선 매칭 로직
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

        is_lane_corrupted = False
        if (final_L >= final_M) or (final_M >= final_R) or (final_R - final_L > ideal_lane_width * 1.5) or (final_R - final_L < ideal_lane_width * 0.5):
            is_lane_corrupted = True

        # 기본 주행 세팅
        base_speed = 20.0
        base_kp = 0.4
        status_text = "DRIVING"
        box_color = (255, 255, 0)
        
        target_x = (final_L + final_R) // 2
        error = target_x - mid_x

        # 3. 정상 차선 추종 제어 및 노란선 양방향 감지
        if not is_lane_corrupted:
            adaptive_kp = base_kp
            if lane_shift_rate > 10 or abs(error) > int(w * 0.05):
                status_text = "ADAPTIVE CURVE MODE"
                box_color = (0, 165, 255)
                adaptive_kp += (lane_shift_rate * 0.02) + (abs(error) * 0.004)
                adaptive_kp = min(adaptive_kp, 1.2)

            calc_angle = float(error * adaptive_kp)
            
            angle_penalty = abs(calc_angle) * 0.32  
            shift_penalty = lane_shift_rate * 0.15
            calc_speed = base_speed - angle_penalty - shift_penalty
            
            if abs(calc_angle) > 15.0 or lane_shift_rate > 12:
                calc_speed = min(calc_speed, 8.0)
            calc_speed = max(7.0, calc_speed)
            
            check_window = 15
            is_L_yellow = np.any(mask_yellow[scan_y, max(0, final_L - check_window):min(w, final_L + check_window)] == 255)
            is_R_yellow = np.any(mask_yellow[scan_y, max(0, final_R - check_window):min(w, final_R + check_window)] == 255)
            
            if is_L_yellow and is_R_yellow:
                status_text = "YELLOW LANE MODE (SLOW)"
                box_color = (0, 255, 255) 
                calc_speed = 5.0
            
            self.prev_L, self.prev_M, self.prev_R = final_L, final_M, final_R
        else:
            status_text = "GUARDRAIL RECOVERY"
            box_color = (0, 0, 255)
            calc_speed = 4.0  
            
            left_weight = np.sum(mask_lane[scan_y:, :mid_x] == 255)
            right_weight = np.sum(mask_lane[scan_y:, mid_x:] == 255)
            
            if right_weight > left_weight * 1.3: calc_angle = -30.0  
            elif left_weight > right_weight * 1.3: calc_angle = 30.0   
            else: calc_angle = float(error * base_kp); calc_speed = 5.5

        # -------------------------------------------------------------------
        # 5. [정지선 판단 알고리즘 선행 연산]
        # -------------------------------------------------------------------
        roi_x_start, roi_x_end = int(w * 0.25), int(w * 0.75)
        roi_y_start, roi_y_end = int(h * 0.70), int(h * 0.80)
        stop_line_roi = mask_white[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        roi_area = (roi_x_end - roi_x_start) * (roi_y_end - roi_y_start)
        white_ratio = np.sum(stop_line_roi == 255) / roi_area

        is_checkerboard = False
        if white_ratio > 0.15: 
            contours, _ = cv2.findContours(stop_line_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_blobs = [c for c in contours if cv2.contourArea(c) > 50]
            if len(valid_blobs) >= 3: is_checkerboard = True

        # -------------------------------------------------------------------
        # 4. 🛰️ [라이다 센서 부 - 인덱스 물리 마스킹 완벽 대교정 완료]
        # -------------------------------------------------------------------
        person_detected = False
        ui_guide_text = ""  
        
        if self.lidar_ranges is not None:
            num_samples = len(self.lidar_ranges)
            
            # 가드레일 간섭 배제용 정면 타겟 50도 화각
            step_25 = int(num_samples * (25.0 / 360.0))
            front_cone_idx = list(range(0, step_25)) + list(range(num_samples - step_25, num_samples))
            valid_indices = [idx for idx in front_cone_idx if 0.25 < self.lidar_ranges[idx] < 2.8]
            
            if self.evade_stage == 0 and len(valid_indices) > 0:
                min_front_dist = np.min(self.lidar_ranges[valid_indices])
                
                # 분기 1: 전방 1.4m 내 돌발 상황 -> 무단횡단 보행자 긴급 제동
                if min_front_dist <= 1.4:
                    person_detected = True
                    print(f"🛑 [돌발 보행자 검출] 정면 {min_front_dist:.2f}m 무단횡단 포착 -> 비상 급제동")
                    cv2.putText(display_img, f"JAYWALKER STOP: {min_front_dist:.2f}m", 
                                (mid_x - 140, int(h * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 3)
                
                # 분기 2: 원거리(1.4m ~ 2.8m) 선제 감지 -> 차량형 장애물 확정 공간 연산
                else:
                    # 💥 [인덱스 대교정]: 자이카 라이다 배열의 반시계 방향 특성 반영
                    step_30 = int(num_samples * (40.0 / 360.0))
                    
                    # 👈 진짜 좌측 30도 공간 (인덱스 0번에서 반시계 방향 증가 구역)
                    left_space = np.sum(self.lidar_ranges[0 : step_30])
                    
                    # 👉 진짜 우측 30도 공간 (인덱스 끝번에서 시계 방향 감소 구역)
                    right_space = np.sum(self.lidar_ranges[num_samples - step_30 : num_samples])
                    
                    if left_space <= 0.1: left_space = 999.0
                    if right_space <= 0.1: right_space = 999.0

                    print("\n" + "="*60)
                    print(f"🚘 [차량 장애물 검출] 전방 거리: {min_front_dist:.2f}m")

                    # 실측된 열린 마진 크기에 기반하여 안전한 방향 확정 분기
                    if left_space > right_space:
                        self.evade_direction = "LEFT"
                        print(f"📢 [터미널 가이드] >>> 진짜 좌측 공간 확보(L:{left_space:.2f}m > R:{right_space:.2f}m). 왼쪽으로 10도 추월! <<<")
                    else:
                        self.evade_direction = "RIGHT"
                        print(f"📢 [터미널 가이드] >>> 진짜 우측 공간 확보(R:{right_space:.2f}m >= L:{left_space:.2f}m). 오른쪽으로 10도 추월! <<<")
                    print("="*60 + "\n")
                    
                    self.evade_stage = 1
                    self.stage_start_time = time.time()

        # ==================================================================
        # ⏱️ [타임라인 오픈루프 회피 제어 시퀀스 - 정규 부호 연동]
        # ==================================================================
        if self.evade_stage == 1:
            # 🚀 1단계 기동: [지정된 방향 10도 회전 + 속도 15 고정 + 1초 동안] 차선 탈출
            status_text = f"⏱️ STAGE 1 ({self.evade_direction} 10 / SPEED 15)"
            ui_guide_text = f"[VEHICLE] {self.evade_direction} OVERTAKE"
            box_color = (0, 140, 255) 
            
            # 자이카 규칙 바인딩: LEFT면 양수(+10.0), RIGHT면 음수(-10.0)
            if self.evade_direction == "LEFT":
                calc_angle = self.fixed_evade_angle
            else:
                calc_angle = -self.fixed_evade_angle
                
            calc_speed = 15.0  
            
            if time.time() - self.stage_start_time >= 3.0:
                print(f"🔄 [STAGE 1 -> 2] 1.0초 완료! 차선 정렬 복귀를 위한 카운터 선회 진입 (1.2초간)")
                self.evade_stage = 2
                self.stage_start_time = time.time()

        elif self.evade_stage == 2:
            # 🔄 2단계 기동: 앞지르기 후 차선 안착 정렬을 위해 정반대로 카운터 조향 10도 유지 (1.2초)
            status_text = f"⏱️ STAGE 2 (COUNTER {self.evade_direction} 10)"
            ui_guide_text = "[VEHICLE] RE-ENTERING..."
            box_color = (255, 0, 255) 
            
            # 카운터 부호 반전: 나갈 때 LEFT(+)였으면 들어올 땐 우회전(-), 나갈 때 RIGHT(-)였으면 들어올 땐 좌회전(+)
            if self.evade_direction == "LEFT":
                calc_angle = -self.fixed_evade_angle
            else:
                calc_angle = self.fixed_evade_angle
                
            calc_speed = 15.0  
            
            if time.time() - self.stage_start_time >= 3.0:
                print(f"✅ [터미널 알림] {self.evade_direction} 10도 / 속도 15 추월 완수! 카메라 차선 주행 복귀.\n")
                self.evade_stage = 0
                self.evade_direction = "NONE"

        # 최종 제어 명령값 바인딩
        if person_detected:
            status_text = "JAYWALKER STOP"
            box_color = (0, 0, 255)
            angle = self.last_normal_angle
            speed = 0.0
            self.evade_stage = 0 
            self.evade_direction = "NONE"
        else:
            angle = max(min(calc_angle, 100.0), -100.0) 
            speed = calc_speed
            self.last_normal_angle = angle
            self.last_normal_speed = speed

        # 정지선 조건 UI 드로잉 최종 처리
        if is_checkerboard:
            status_text = "CHECKERBOARD (PASS)"; box_color = (255, 0, 255) 
        elif white_ratio >= 0.70:
            status_text = f"STOP LINE (Ratio: {white_ratio*100:.1f}%)"
            box_color = (0, 0, 255); speed, angle = 0.0, 0.0
            self.evade_stage = 0
            self.evade_direction = "NONE"
        else:
            if not person_detected:
                cv2.putText(display_img, f"White Ratio: {white_ratio*100:.1f}%", (roi_x_start, roi_y_start - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 6. 실시간 모니터링 디스플레이 그리기
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

        if ui_guide_text != "":
            cv2.putText(display_img, ui_guide_text, (mid_x - 140, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 140, 255), 2)

        cv2.putText(display_img, f"Status: {status_text}", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        cv2.putText(display_img, f"Angle: {angle:.1f} | Speed: {speed:.1f}", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Lane & Stopline Tracking", display_img)
        cv2.waitKey(1)

        # 7. 제어 토픽 퍼블리시
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
        pass
    finally:
        stop_msg = XycarMotor()
        stop_msg.angle = 0.0; stop_msg.speed = 0.0
        node.motor_pub.publish(stop_msg)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
