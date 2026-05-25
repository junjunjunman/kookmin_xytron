#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrafficLightTestNode(Node):
    def __init__(self):
        super().__init__('traffic_light_test_node')
        self.get_logger().info('==========================================')
        self.get_logger().info('=== 🚥 신호등 인식 알고리즘 독립 검증 노드 시작 ===')
        self.get_logger().info(' - 차량 제어(출발) 없이 오직 탐색력만 실시간 테스트합니다.')
        self.get_logger().info(' - 초록불 점등 시 함수가 True를 반환하는지 검증하세요.')
        self.get_logger().info('==========================================')
        
        self.bridge = CvBridge()
        self.image = None
        
        # 전방 카메라 토픽 구독 설정 [cite: 6]
        self.sub_front = self.create_subscription(
            Image, 
            '/usb_cam/image_raw/front', 
            self.cam_callback, 
            qos_profile_sensor_data
        )
        
        # 0.1초(10Hz) 주기로 신호등 판단 함수를 반복 실행하는 안전 제어 타이머 생성 [cite: 22]
        self.timer = self.create_timer(0.1, self.timer_callback)

    def cam_callback(self, data):
        """ 실시간 전방 카메라 토픽 수신 """
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")

    def timer_callback(self):
        """ 타이머 루프에서 검증 대상 함수를 호출하고 리턴값을 감시 """
        if self.image is None:
            return
            
        # 검증 대상 함수 실행 
        is_green = self.check_3_light_green()
        
        # 최종 리턴값에 따른 가시적인 결과 로그 스트리밍 [cite: 44]
        if is_green:
            self.get_logger().info("🔥 [함수 리턴 결과] check_3_light_green() -> True 반환! (메인 코드였다면 차량 출발)")
        else:
            self.get_logger().info("💤 [함수 리턴 결과] check_3_light_green() -> False 반환! (대기 중)")

    # =======================================================
    # [검증 대상 함수] 사용자가 구현한 로직 그대로 이식
    # =======================================================
    def check_3_light_green(self):
        """ 미션 1: 3구 신호등이 녹색(파란불)인지 판별 """
        if self.image is None:
            return False

        # 1. 사용자가 마우스로 획득한 픽셀 좌표 기반 ROI 지정 
        # Numpy Array 슬라이싱 구조: [Y_시작:Y_끝, X_시작:X_끝] [cite: 48]
        roi = self.image[87:144, 336:393]

        # 2. 조명 변화에 강건한 HSV 색 공간 변환 [cite: 4]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 3. 초록색(파란불) 불빛의 HSV 검출 범위 정의 [cite: 4]
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        # 4. 범위 안의 픽셀만 흰색(255)으로 걸러내는 마스크 생성 [cite: 4]
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 5. 마스크 내부의 흰색 픽셀(녹색 신호) 개수 카운트 [cite: 4]
        green_pixel_count = cv2.countNonZero(mask)

        # =======================================================
        # [실시간 디버깅 창] 신호가 잘 잡히는지 눈으로 확인하는 뷰어
        # =======================================================
        cv2.imshow("Traffic Light ROI", roi)
        cv2.imshow("Green Filter Mask", mask)
        cv2.waitKey(1)

        # 6. 기준 픽셀 수(Threshold) 초과 시 True 리턴하여 출발 신호 발생
        # 총 픽셀 수(57 * 57 = 3249) 중 불빛이 들어왔을 때의 임계값을 300으로 설정 
        if green_pixel_count > 300:
            self.get_logger().info(f"🟢 [신호등 내부] 초록불 확인! 픽셀 카운트: {green_pixel_count}")
            return True
        else:
            # 픽셀 수 추이를 편하게 보기 위해 엘스문 디버깅 로그 추가
            if green_pixel_count > 0:
                self.get_logger().info(f"🔴 [신호등 내부] 빨간불/노란불 상태 (초록색 픽셀 수: {green_pixel_count}/300)")
        
        return False

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
