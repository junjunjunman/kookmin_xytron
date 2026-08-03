#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy # LiDAR QoS 설정을 위해 추가[cite: 2]
from sensor_msgs.msg import Image, LaserScan        # LaserScan 메시지 타입 추가[cite: 1, 2]
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================================
# 🛠️ 이전에 작성된 콘 주행 코드를 같은 폴더에 cone_drive.py로 저장해두고 불러옵니다.
from cone_drive import start_cone_mode 
# ==========================================================

class CamNode(Node):
    def __init__(self):
        super().__init__('cam_node')
        self.bridge = CvBridge()
        
        # 1. YOLO 모델 로드[cite: 1]
        model_path = '/home/xytron/xycar_ws/src/drive/best.pt'
        self.get_logger().info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        # 2. ROS2 퍼블리셔 / 서브스크라이버 설정 (카메라)[cite: 1]
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Float32MultiArray, 'drive_cmd', 10)

        # ==========================================
        # ⭐ 신규 추가: LiDAR 서브스크라이버 설정[cite: 2]
        # ==========================================
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)[cite: 2]
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',  # 실제 환경에 맞게 'scan' 또는 '/scan' 확인 필요
            self.lidar_callback,
            qos_profile
        )[cite: 2]
        self.current_lidar_ranges = None # 실시간 라이다 데이터를 저장할 변수[cite: 2]
        
        # ==========================================
        # 파라미터 및 설정 변수[cite: 1]
        # ==========================================
        self.WIDTH = 640
        self.HEIGHT = 480
        
        # ROI 영역 (0.05 ~ 0.95, 0.60 ~ 0.99)[cite: 1]
        self.roi_x_start = int(self.WIDTH * 0.05)
        self.roi_x_end = int(self.WIDTH * 0.95)
        self.roi_y_start = int(self.HEIGHT * 0.60)
        self.roi_y_end = int(self.HEIGHT * 0.70)
        
        self.USE_BLACK_ROI = True[cite: 1]
        
        self.FIXED_SPEED = 5.0[cite: 1]
        self.MAX_ANGLE = 40.0[cite: 1]
        
        # PID 제어 변수[cite: 1]
        self.kp = 0.4
        self.ki = 0.0002
        self.kd = 0.75
        self.integral = 0.0
        self.prev_error = 0.0
        self.max_integral = 500.0

        # 트리거 인지용 픽셀 임계값 및 상태 플래그
        self.CONE_AREA_THRESHOLD = 6000
        self.is_cone_driving_active = False 

        self.get_logger().info("Autonomous Driving Node (Camera + LiDAR Ready!)")

    # ==========================================
    # ⭐ 신규 추가: LiDAR 데이터 수신 콜백[cite: 2]
    # ==========================================
    def lidar_callback(self, msg):
        """LiDAR 데이터를 저장하는 콜백 함수"""
        # 콘 주행 로직에서 사용하기 위해 1~504 인덱스 슬라이싱 값을 상시 업데이트[cite: 2]
        self.current_lidar_ranges = msg.ranges[1:505]

    def image_callback(self, msg):
        # 1. 이미지 변환[cite: 1]
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. YOLO 추론[cite: 1]
        results = self.model.predict(source=cv_image, save=False, conf=0.25, verbose=False)
        result = results[0]
        
        # 3. 마스크 생성 및 콘 면적 확인
        lane_mask = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)[cite: 1]
        max_cone_area = 0

        if result.masks is not None and result.boxes is not None:[cite: 1]
            masks = result.masks.data.cpu().numpy()[cite: 1]
            classes = result.boxes.cls.cpu().numpy()[cite: 1]
            
            for mask, cls_id in zip(masks, classes):[cite: 1]
                class_name = result.names[int(cls_id)][cite: 1]
                mask_resized = cv2.resize(mask, (self.WIDTH, self.HEIGHT))[cite: 1]
                
                # 기존 차선 (left, right) 필터링[cite: 1]
                if class_name.lower() in ['left', 'right']:
                    lane_mask[mask_resized > 0.5] = 255[cite: 1]
                
                # 콘 객체의 가장 큰 픽셀 넓이 탐색
                elif class_name.lower() == 'cone':
                    area = int(np.sum(mask_resized > 0.5))
                    if area > max_cone_area:
                        max_cone_area = area
        
        # ==========================================
        # 4. 주행 모드 스위치 로직
        # ==========================================
        # 아직 기본 주행 중인데, 임계값보다 큰 콘이 나타나면? -> 콘 주행 스위치 ON
        if max_cone_area > self.CONE_AREA_THRESHOLD and not self.is_cone_driving_active:
            self.get_logger().info(f"🚨 콘 트리거 인지됨! (크기: {max_cone_area}px). 콘 사이 주행 모드로 진입합니다.")
            self.is_cone_driving_active = True 

        # 현재 어느 주행 모드인지에 따라 함수 분기
        if self.is_cone_driving_active:
            # 🚧 외부 콘 주행 모듈 호출 (카메라 + 라이다 데이터 함께 전달)
            angle, debug_img, is_finished = start_cone_mode(
                self, cv_image, result, self.current_lidar_ranges
            )
            
            # 외부 모듈에서 left/right가 안보인다고(종료 신호) 알려오면? -> 기본 주행 스위치 전환
            if is_finished:
                self.get_logger().info("✅ 콘 주행 모드 종료! 다음 프레임부터 기본 주행으로 복귀합니다.")
                self.is_cone_driving_active = False 
                angle = 0.0 # 전환 시 갑자기 핸들이 꺾이는 것을 막기 위해 중앙 정렬
                
        else:
            # 🛣️ 평상시 기본 차선 주행 모드 수행
            angle, debug_img = self.process_vision_and_control(lane_mask, cv_image)[cite: 1]
        
        # 5. 제어 명령 퍼블리시 및 뷰어 출력
        drive_msg = Float32MultiArray()[cite: 1]
        drive_msg.data = [float(angle), float(self.FIXED_SPEED)][cite: 1]
        self.publisher.publish(drive_msg)[cite: 1]
        
        cv2.imshow("Autonomous Driving View", debug_img)[cite: 1]
        cv2.waitKey(1)[cite: 1]

    # ==========================================
    # ⚠️ 기존 기본 주행 로직 (절대 수정하지 않음)[cite: 1]
    # ==========================================
    def process_vision_and_control(self, mask, img):[cite: 1]
        out_img = img.copy()[cite: 1]
        
        roi_mask = mask[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end][cite: 1]
        
        if self.USE_BLACK_ROI:[cite: 1]
            out_img[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end] = (0, 0, 0)[cite: 1]
            roi_view = out_img[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end][cite: 1]
            roi_view[roi_mask == 255] = (255, 255, 255)[cite: 1]
        else:[cite: 1]
            pass[cite: 1]
            
        cv2.rectangle(out_img, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), (0, 0, 0), 2)[cite: 1]
        
        roi_height = self.roi_y_end - self.roi_y_start[cite: 1]
        nwindows = 4[cite: 1]
        window_height = int(roi_height / nwindows)[cite: 1]
        margin = 40[cite: 1]
        minpix = 30[cite: 1]
        
        roi_mask_full = np.zeros_like(mask)[cite: 1]
        roi_mask_full[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end] = roi_mask[cite: 1]
        
        nonzero = roi_mask_full.nonzero()[cite: 1]
        nonzeroy = np.array(nonzero[0])[cite: 1]
        nonzerox = np.array(nonzero[1])[cite: 1]
        
        histogram = np.sum(roi_mask_full[self.roi_y_start + roi_height//2 : self.roi_y_end, :], axis=0)[cite: 1]
        midpoint = int(histogram.shape[0] / 2)[cite: 1]
        
        leftx_base = np.argmax(histogram[:midpoint])[cite: 1]
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint[cite: 1]
        
        if leftx_base == 0: leftx_base = self.roi_x_start + 50[cite: 1]
        if rightx_base == midpoint: rightx_base = self.roi_x_end - 50[cite: 1]
        
        leftx_current = leftx_base[cite: 1]
        rightx_current = rightx_base[cite: 1]
        
        for window in range(nwindows):[cite: 1]
            win_y_low = self.roi_y_end - (window + 1) * window_height[cite: 1]
            win_y_high = self.roi_y_end - window * window_height[cite: 1]
            
            win_xleft_low = leftx_current - margin[cite: 1]
            win_xleft_high = leftx_current + margin[cite: 1]
            win_xright_low = rightx_current - margin[cite: 1]
            win_xright_high = rightx_current + margin[cite: 1]
            
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)[cite: 1]
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)[cite: 1]
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &[cite: 1]
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0][cite: 1]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &[cite: 1]
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0][cite: 1]
                               
            if len(good_left_inds) > minpix:[cite: 1]
                leftx_current = int(np.mean(nonzerox[good_left_inds]))[cite: 1]
            if len(good_right_inds) > minpix:[cite: 1]
                rightx_current = int(np.mean(nonzerox[good_right_inds]))[cite: 1]

        sw_final_M = int((leftx_current + rightx_current) / 2)[cite: 1]
        
        mid_x = self.WIDTH // 2[cite: 1]
        error = sw_final_M - mid_x[cite: 1]
        
        if error * self.prev_error <= 0:[cite: 1]
            self.integral = 0.0[cite: 1]
            
        p_term = self.kp * error[cite: 1]
        self.integral = max(min(self.integral + error, self.max_integral), -self.max_integral)[cite: 1]
        i_term = self.ki * self.integral[cite: 1]
        d_term = self.kd * (error - self.prev_error)[cite: 1]
        
        self.prev_error = error[cite: 1]
        calc_angle = p_term + i_term + d_term[cite: 1]
        
        calc_angle = max(min(calc_angle, self.MAX_ANGLE), -self.MAX_ANGLE)[cite: 1]
        
        cv2.circle(out_img, (mid_x, self.roi_y_start + roi_height//2), 5, (255, 0, 0), -1)[cite: 1]
        cv2.circle(out_img, (sw_final_M, self.roi_y_start + roi_height//2), 5, (0, 0, 255), -1)[cite: 1]
        cv2.line(out_img, (mid_x, self.roi_y_end), (sw_final_M, self.roi_y_start), (0, 255, 255), 2)[cite: 1]
        
        return calc_angle, out_img[cite: 1]

def main(args=None):
    rclpy.init(args=args)[cite: 1]
    node = CamNode()[cite: 1]
    try:
        rclpy.spin(node)[cite: 1]
    except KeyboardInterrupt:[cite: 1]
        pass[cite: 1]
    finally:
        node.destroy_node()[cite: 1]
        rclpy.shutdown()[cite: 1]
        cv2.destroyAllWindows()[cite: 1]

if __name__ == '__main__':
    main()[cite: 1]