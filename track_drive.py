#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
import rclpy, time, cv2, os, math
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from cv_bridge import CvBridge

#=============================================
# ROS2 Node 클래스 정의
#=============================================
class TrackDriverNode(Node):

    #=============================================
    # 클래스 생성 초기화 함수
    #=============================================
    def __init__(self):
        super().__init__('driver')
        self.get_logger().info('----- Xycar self-driving node started -----')
        
        # 상수값 및 초기값 설정
        self.image = None              # 카메라 토픽 데이터를 저장할 변수
        self.motor_msg = XycarMotor()  # 모터토픽 메시지        
        self.lidar_ranges = None       # 라이다 토픽 데이터를 저장할 변수
        self.bridge = CvBridge()
        
        # [확장] 주행 상태 관리를 위한 변수 선언 (디버깅 및 수정 용이)
        self.current_lap = 1           # 현재 바퀴 수 (1, 2, 3바퀴)
        self.current_state = "WAIT_3_LIGHT"  # 현재 차량의 주행 상태 (상태 머신)
        
        # ROS2 Publisher & Subscriber 설정
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.cam_callback, qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        
        self.get_logger().info("Track Driver Node Initialized")
              
    #=============================================
    # 카메라 토픽을 수신하는 콜백 함수
    #=============================================
    def cam_callback(self, data):
        # 수신한 메시지를 OpenCV 이미지로 변환하여 저장
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    #=============================================
    # 라이다 토픽을 수신하는 콜백 함수
    #=============================================
    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges   
      
    #=============================================
    # 모터제어 토픽을 발행하는 Publisher 함수 (안전장치 포함)
    #=============================================
    def drive(self, angle, speed):
        # 조작 패널 사양 및 차량 보호를 위한 값 제한 (Clamp)
        # 조향각 제한: -100 ~ 100, 속도 제한: -50 ~ 50
        bounded_angle = max(-100.0, min(100.0, float(angle)))
        bounded_speed = max(-50.0, min(50.0, float(speed)))

        self.motor_msg.angle = bounded_angle
        self.motor_msg.speed = bounded_speed
        self.motor_pub.publish(self.motor_msg)

    #=======================================================
    # [설계] 인지(Perception) 및 판단 함수 정의 파트
    #=======================================================
    
    def check_3_light_green(self):
        """ 미션 1: 3구 신호등이 녹색(파란불)인지 판별 """
        if self.image is None:
            return False

        # 1. 사용자가 마우스로 획득한 픽셀 좌표 기반 ROI 지정 
        # Numpy Array 슬라이싱 구조: [Y_시작:Y_끝, X_시작:X_끝]
        roi = self.image[87:144, 336:393]

        # 2. 조명 변화에 강건한 HSV 색 공간 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 3. 초록색(파란불) 불빛의 HSV 검출 범위 정의
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        # 4. 범위 안의 픽셀만 흰색(255)으로 걸러내는 마스크 생성
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 5. 마스크 내부의 흰색 픽셀(녹색 신호) 개수 카운트
        green_pixel_count = cv2.countNonZero(mask)

        # =======================================================
        # [실시간 디버깅 창] 신호가 잘 잡히는지 눈으로 확인하는 뷰어
        # 대회 도중 시각화가 불필요해지면 아래 3줄은 주석 처리.
        # =======================================================
        # cv2.imshow("Traffic Light ROI", roi)
        cv2.imshow("Green Filter Mask", mask)
        cv2.waitKey(1)

        # 6. 기준 픽셀 수(Threshold) 초과 시 True 리턴하여 출발 신호 발생
        # 총 픽셀 수(57 * 57 = 3249) 중 불빛이 들어왔을 때의 임계값을 300으로 설정
        if green_pixel_count > 300:
            self.get_logger().info(f"🟢 초록불 픽셀 수: {green_pixel_count} -> 출발!")
            return True
        
        return False

    def process_lidar_cones(self):
        """ 미션 2: 라바콘 사이를 주행하기 위한 조향각 및 속도 계산 """
        # TODO: self.lidar_ranges 정보를 파싱하여 좌/우 라바콘 중심점 추종
        angle = 0.0
        speed = 5.0
        return angle, speed

    def process_lane_following(self):
        """ 미션 3, 5, 6: 아스팔트 차선 인식 및 곡률 기반 주행 제어 """
        # TODO: OpenCV 차선 인식을 통해 직선 구간(가속), 지그재그(감속 및 차선 허용) 제어
        angle = 0.0
        speed = 8.0  # 직선구간 가속 및 지그재그구간 속도 가변 제어 필요
        return angle, speed

    def check_4_light_and_police(self):
        """ 미션 4: 4구 신호등 상태 및 지름길 경찰차 유무 판별 """
        # TODO: 4구 신호등 화살표(좌회전) 인식 및 카메라/라이다로 경찰차 방해물 체크
        # 반환값 예시: "STRAIGHT" (직진해야함), "SHORTCUT" (지름길 진입가능)
        return "STRAIGHT"

    def process_shortcut(self):
        """ 미션 4-지름길: 장애물이 떨어지는 유동적 차선 주행 """
        # TODO: 차선이 유실되거나 방해물이 떨어질 때의 예외 주행 알고리즘
        angle = 0.0
        speed = 6.0
        return angle, speed

    def process_avoidance_and_school(self, speed):
        """ 미션 5, 6, 7: 보행자 회피, 방해차량 추월, 어린이 보호구역 통합 제어 """
        # TODO: 주행 중 라이다 감지 시 보행자/차량 회피 및 바닥 인식을 통한 속도 제어
        # 이 함수는 계산된 기본 speed를 입력받아 최종 안전 speed를 리턴 조절하도록 설계
        final_speed = speed
        return final_speed

    def is_checkerboard_detected(self):
        """ 바퀴 수 누적을 위한 체커보드 마커 감지 """
        # TODO: 카메라 하단 영역에서 체커보드 패턴 혹은 특정 마커 인식 시 True
        return False

    #=============================================
    # 메인 루프 (상태 머신 구동 및 데이터 갱신)
    #=============================================
    def main_loop(self):
    
        self.get_logger().info("======================================")
        self.get_logger().info("   S T A R T   D R I V I N G ...      ")
        self.get_logger().info("======================================")

        while rclpy.ok():
            # [★가장 중요★] 루프 안에서 spin_once를 호출해야 센서 콜백 데이터가 실시간 갱신됩니다!
            rclpy.spin_once(self, timeout_sec=0.01)

            # 초기 데이터 미수신 상태 예외 처리 (None 에러 방지)
            if self.image is None or self.lidar_ranges is None:
                continue

            #---------------------------------------------------
            # 공통 매커니즘: 바퀴 수 완료 체크 (체커보드 인식)
            #---------------------------------------------------
            if self.is_checkerboard_detected():
                self.current_lap += 1
                self.get_logger().info(f"체커보드 통과! 현재 바퀴 수: {self.current_lap}/3")
                
                if self.current_lap > 3:
                    self.current_state = "FINISHED"
                else:
                    # 2, 3바퀴째는 3구 신호등과 라바콘이 없으므로 바로 차선주행 상태로 리셋
                    self.current_state = "LANE_DRIVE"
                time.sleep(0.5) # 중복 인식 방지 디레이

            #---------------------------------------------------
            # 핵심 자율주행 상태 머신 (State Machine)
            #---------------------------------------------------
            target_angle = 0.0
            target_speed = 0.0

            if self.current_state == "WAIT_3_LIGHT":
                # 미션 1: 멈춘 상태로 3구 신호등 대기
                self.get_logger().info("상태: 3구 신호등 파란불 대기 중...", once=True)
                if self.check_3_light_green():
                    self.get_logger().info("파란불 점등! 라바콘 구간으로 진입합니다.")
                    self.current_state = "CONE_DRIVE"
                else:
                    target_angle, target_speed = 0.0, 0.0

            elif self.current_state == "CONE_DRIVE":
                # 미션 2: 라바콘 곡선 구간 통과 (1바퀴째 전용)
                target_angle, target_speed = self.process_lidar_cones()
                
                # 라바콘 구간 탈출 조건 (예: 전방 라바콘 청소 완료 및 아스팔트 감지 시)
                # 실전 테스트 후 조건 만족 시 상태 변경 작성: self.current_state = "LANE_DRIVE"
                pass 

            elif self.current_state == "LANE_DRIVE":
                # 미션 3, 5, 6, 7: 일반 아스팔트 주행 및 부가 미션 대응
                target_angle, target_speed = self.process_lane_following()
                
                # 안전 미션(보행자, 추월, 어린이보호구역)에 따른 속도 가변 재가공
                target_speed = self.process_avoidance_and_school(target_speed)

                # 4구 신호등 교차로 부근에 도달했을 때 분기 판단
                if self.current_lap in [2, 3]:
                    decision = self.check_4_light_and_police()
                    if decision == "SHORTCUT":
                        self.get_logger().info("지름길 진입 조건을 만족하여 회전합니다!")
                        self.current_state = "SHORTCUT_DRIVE"

            elif self.current_state == "SHORTCUT_DRIVE":
                # 미션 4: 2, 3바퀴째 지름길 전용 특수 주행
                target_angle, target_speed = self.process_shortcut()

            elif self.current_state == "FINISHED":
                # 3바퀴 최종 완주 상태
                self.get_logger().info("모든 트랙 완주! 차량을 정지합니다.")
                target_angle, target_speed = 0.0, 0.0
                self.drive(target_angle, target_speed)
                break

            # 최종 가공된 조향각과 속도로 차량 구동
            self.drive(target_angle, target_speed)

#=============================================
# 메인 함수
#=============================================
def main(args=None):
      
    rclpy.init(args=args)
    node = TrackDriverNode()
    
    try:
        # main_loop() 함수를 호출하여 실행합니다.
        node.main_loop()
    except KeyboardInterrupt:
        # 사용자 인터럽트 (Ctrl+C)가 발생하면 예외를 처리합니다.
        pass
    finally:
        # 노드를 종료하고 ROS2를 정리합니다.
        node.drive(angle=0, speed=0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()