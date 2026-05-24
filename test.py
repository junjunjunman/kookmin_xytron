#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2

class TrafficLightTunerNode(Node):
    def __init__(self):
        super().__init__('traffic_light_tuner')
        self.get_logger().info('==========================================')
        self.get_logger().info('=== 🚥 신호등 좌표 탐색 노드 (test.py) 시작 ===')
        self.get_logger().info(' 1. 팝업 창의 신호등 위치를 마우스로 클릭하세요.')
        self.get_logger().info(' 2. 터미널에 출력되는 X, Y 좌표를 확인하세요.')
        self.get_logger().info('==========================================')
        
        self.bridge = CvBridge()
        
        # -------------------------------------------------------------
        # [튜닝 포인트] 방법 1로 마우스 클릭 좌표를 알아낸 뒤, 아래 값을 수정해 보세요!
        # 수정 후 코드를 다시 실행하면 초록색 네모 박스가 해당 위치에 live로 그려집니다.
        # -------------------------------------------------------------
        self.roi_x1 = 200  # 좌상단 X
        self.roi_y1 = 50   # 좌상단 Y
        self.roi_x2 = 400  # 우하단 X
        self.roi_y2 = 150  # 우하단 Y

        # 전방 카메라 토픽 구독 설정
        self.sub_front = self.create_subscription(
            Image, 
            '/usb_cam/image_raw/front', 
            self.cam_callback, 
            qos_profile_sensor_data
        )

    def on_mouse_click(self, event, x, y, flags, param):
        """ 마우스 왼쪽 버튼 클릭 시 터미널에 좌표를 찍어주는 콜백 함수 """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.get_logger().info(f"🎯 [클릭 좌표] X: {x}, Y: {y}")

    def cam_callback(self, data):
        try:
            # ROS Image 메시지를 OpenCV BGR 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return

        # 원본 이미지 보호 및 사각형 시각화를 위해 화면용 이미지 복사
        display_img = cv_image.copy()

        # OpenCV 윈도우 창 생성 및 마우스 이벤트 연결
        window_name = "Traffic Light Tuner"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse_click)

        # 설정된 ROI 영역을 화면에 초록색(0, 255, 0) 사각형으로 live 표시 (두께: 2)
        cv2.rectangle(
            display_img, 
            (self.roi_x1, self.roi_y1), 
            (self.roi_x2, self.roi_y2), 
            (0, 255, 0), 
            2
        )
        
        # 사각형 상단에 식별 텍스트 출력
        cv2.putText(
            display_img, 
            "ROI Area Preview", 
            (self.roi_x1, self.roi_y1 - 7), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            1
        )

        # 최종 가공화면 출력
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightTunerNode()
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