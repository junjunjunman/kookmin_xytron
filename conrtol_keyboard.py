#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import sys
import termios
import tty

# 터미널에 표시될 조작 안내 메시지
msg = """
=================================
  Xycar Keyboard Controller
=================================
  w : 속도 증가 (전진)
  s : 속도 감소 (후진)
  a : 왼쪽 조향
  d : 오른쪽 조향
  x : 정지 (속도 0, 조향각 0)
  
  Ctrl-C 누르면 종료
=================================
"""

class KeyboardController(Node):
    def __init__(self):
        super().__init__('keyboard_controller')
        # hough_drive.py와 동일하게 xycar_motor 토픽으로 발행
        self.motor_publisher = self.create_publisher(Float32MultiArray, 'xycar_motor', 1)
        
        self.speed = 0.0
        self.angle = 0.0
        
        # 차량 제어 한계값 설정 (안전을 위해 제한)
        self.MAX_SPEED = 50.0
        self.MAX_ANGLE = 50.0

    def publish_motor(self):
        """현재 조향각과 속도를 토픽으로 발행"""
        motor_msg = Float32MultiArray()
        motor_msg.data = [float(self.angle), float(self.speed)]
        self.motor_publisher.publish(motor_msg)

    def clamp_values(self):
        """속도와 조향각이 안전 범위를 넘지 않도록 제한"""
        self.speed = max(min(self.speed, self.MAX_SPEED), -self.MAX_SPEED)
        self.angle = max(min(self.angle, self.MAX_ANGLE), -self.MAX_ANGLE)

def get_key(settings):
    """엔터키 입력 없이 실시간으로 키보드 1바이트 입력을 받는 함수"""
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main(args=None):
    # 현재 터미널 설정 저장
    settings = termios.tcgetattr(sys.stdin)
    
    rclpy.init(args=args)
    node = KeyboardController()
    
    print(msg)
    
    try:
        while rclpy.ok():
            key = get_key(settings)
            
            if key == 'w':
                node.speed += 5.0
            elif key == 's':
                node.speed -= 5.0
            elif key == 'a':
                node.angle -= 10.0
            elif key == 'd':
                node.angle += 10.0
            elif key == 'x':
                node.speed = 0.0
                node.angle = 0.0
            elif key == '\x03': # Ctrl-C
                break
            
            # 한계치 필터링 및 모터 명령 발행
            node.clamp_values()
            node.publish_motor()
            
            # 현재 상태를 터미널 창 같은 줄에 덮어쓰기로 출력
            print(f"\r현재 상태 ➔ Angle: {node.angle:>5.1f} | Speed: {node.speed:>5.1f}   ", end='')

    except Exception as e:
        print(f"\n오류 발생: {e}")
        
    finally:
        # 종료 시 차량을 멈추기 위한 안전 로직
        print("\n프로그램을 종료하며 차량을 정지합니다.")
        node.speed = 0.0
        node.angle = 0.0
        node.publish_motor()
        
        # 터미널 설정을 원래대로 복구
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()