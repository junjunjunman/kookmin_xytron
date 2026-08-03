import rclpy
from rclpy.node import Node
import numpy as np
import cv2

# ==========================================================
# [설명]
# 이 함수는 CamNode(기본 주행 코드) 내부에서 호출되어 콘 사이를 주행합니다.
# YOLO 인스턴스(left, right, current)와 504등분 라이다 인덱스를 연동하고,
# 경로의 평균 기울기를 계산하여 부드러운 조향을 수행합니다.
# ==========================================================

def start_cone_mode(cam_node_instance, cv_image, yolo_results, lidar_ranges):
    """
    콘 사이 주행(Cone Driving Mode) 로직.
    
    :param cam_node_instance: CamNode 클래스의 인스턴스
    :param cv_image: 현재 카메라 이미지(Matplotlib 렌더링 화면) 프레임
    :param yolo_results: YOLO 모델 예측 결과 객체
    :param lidar_ranges: 현재 LiDAR 스캔 데이터 (ranges[1:505])
    :return: calc_angle (조향각), debug_img (시각화 이미지), is_finished (모드 종료 여부)
    """
    debug_img = cv_image.copy()[cite: 4, 6]
    is_finished = False[cite: 4, 6]
    calc_angle = 0.0[cite: 4, 6]
    
    H, W = debug_img.shape[:2][cite: 6]
    SLICE_STEP = (2 * np.pi) / 504  # 360도를 504등분한 라디안 값[cite: 6]

    # ---------------------------------------------------------
    # 1. YOLO 객체 탐지 및 좌표 추출 (current, left, right)
    # ---------------------------------------------------------
    current_points = [][cite: 6]
    left_boxes = [][cite: 6]
    right_boxes = [][cite: 6]
    left_centroids = [][cite: 6]
    right_centroids = [][cite: 6]

    if yolo_results.boxes is not None:[cite: 4, 6]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()[cite: 4, 6]
        classes = yolo_results.boxes.cls.cpu().numpy()[cite: 4, 6]
        names = yolo_results.names[cite: 4, 6]

        for box, cls_id in zip(boxes, classes):[cite: 4, 6]
            class_name = str(names[int(cls_id)]).lower()[cite: 4, 6]
            x1, y1, x2, y2 = map(int, box)[cite: 6]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)[cite: 6]
            
            if 'current' in class_name:[cite: 6]
                conf = float(yolo_results.boxes.conf[np.where(classes == cls_id)[0][0]].cpu().numpy())[cite: 6]
                current_points.append((x1, y1, x2, y2, conf))[cite: 6]
                cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)[cite: 6]
            elif 'left' in class_name:[cite: 6]
                left_boxes.append((x1, y1, x2, y2))[cite: 6]
                left_centroids.append((cx, cy))[cite: 6]
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 2)[cite: 6]
            elif 'right' in class_name:[cite: 6]
                right_boxes.append((x1, y1, x2, y2))[cite: 6]
                right_centroids.append((cx, cy))[cite: 6]
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)[cite: 6]

    # 🚨 종료 조건: left와 right가 모두 감지되지 않을 때 기본 주행 복귀
    if len(left_boxes) == 0 and len(right_boxes) == 0:[cite: 4, 6]
        cam_node_instance.get_logger().info("콘(left/right)이 감지되지 않아 기본 주행으로 복귀합니다.")[cite: 4, 6]
        is_finished = True[cite: 4, 6]
        return 0.0, debug_img, is_finished[cite: 4, 6]

    # ---------------------------------------------------------
    # 2. current_point 기준점 (Cx, Cy) 설정
    # ---------------------------------------------------------
    if current_points:[cite: 6]
        current_points.sort(key=lambda x: x[4], reverse=True)[cite: 6]
        bx1, by1, bx2, by2, _ = current_points[0][cite: 6]
        Cx = int((bx1 + bx2) / 2)[cite: 6]
        Cy = int((by1 + by2) / 2)[cite: 6]
    else:[cite: 6]
        Cx, Cy = W // 2, H[cite: 6]

    cv2.drawMarker(debug_img, (Cx, Cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)[cite: 6]

    # ---------------------------------------------------------
    # 3. 중앙 안전 경로 생성 및 💡선분 각도 평균(Look-ahead) 계산
    # ---------------------------------------------------------
    # 로봇과 가까운 것부터 정렬 (Y축 기준)
    left_centroids.sort(key=lambda x: x[1], reverse=True)[cite: 6]
    right_centroids.sort(key=lambda x: x[1], reverse=True)[cite: 6]

    central_path = [][cite: 6]
    num_pairs = min(len(left_centroids), len(right_centroids))[cite: 6]

    for i in range(num_pairs):[cite: 6]
        L = left_centroids[i][cite: 6]
        R = right_centroids[i][cite: 6]
        cv2.line(debug_img, L, R, (0, 165, 255), 1)[cite: 6] # 주황색 횡단선
        
        mid_x = int((L[0] + R[0]) / 2)[cite: 6]
        mid_y = int((L[1] + R[1]) / 2)[cite: 6]
        central_path.append((mid_x, mid_y))[cite: 6]
        cv2.circle(debug_img, (mid_x, mid_y), 4, (0, 255, 255), -1)[cite: 6]

    # 노란색 경로 선 긋기
    if len(central_path) > 1:[cite: 6]
        for i in range(len(central_path) - 1):[cite: 6]
            cv2.line(debug_img, central_path[i], central_path[i+1], (0, 255, 255), 3)[cite: 6]

    # 각 구간의 기울기(각도)를 구해 평균 산출 (부드러운 조향)
    if len(central_path) > 0:[cite: 6]
        path_points = [(Cx, Cy)] + central_path[cite: 6]
        segment_angles = [][cite: 6]
        
        for i in range(len(path_points) - 1):[cite: 6]
            p1 = path_points[i][cite: 6]
            p2 = path_points[i+1][cite: 6]
            
            dx = p2[0] - p1[0][cite: 6]
            dy = p1[1] - p2[1] # Y좌표는 위로 갈수록 작아짐[cite: 6]
            dy = max(1, dy)[cite: 6]
            
            angle_deg = np.degrees(np.arctan2(dx, dy))[cite: 6]
            segment_angles.append(angle_deg)[cite: 6]
            
        steering_angle = float(np.mean(segment_angles))[cite: 6]
    else:[cite: 6]
        # 한쪽 콘만 감지되었을 때의 예외 처리
        if left_centroids:[cite: 6]
            target_x = left_centroids[0][0] + 200[cite: 6]
        elif right_centroids:[cite: 6]
            target_x = right_centroids[0][0] - 200[cite: 6]
        else:[cite: 6]
            target_x = Cx[cite: 6]
        steering_angle = float(np.degrees(np.arctan2(target_x - Cx, 150)))[cite: 6]

    # ---------------------------------------------------------
    # 4. 피자 조각 인덱스 추출 및 LiDAR 충돌 회피 보정
    # ---------------------------------------------------------
    def get_angle(x, y):[cite: 6]
        return np.arctan2(Cx - x, Cy - y)[cite: 6]

    left_angles = [][cite: 6]
    for (x1, y1, x2, y2) in left_boxes:[cite: 6]
        left_angles.extend([get_angle(x, y) for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]])[cite: 6]
    
    right_angles = [][cite: 6]
    for (x1, y1, x2, y2) in right_boxes:[cite: 6]
        right_angles.extend([get_angle(x, y) for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]])[cite: 6]

    max_left_angle = max(left_angles) if left_angles else np.pi / 2[cite: 6]
    min_right_angle = min(right_angles) if right_angles else -np.pi / 2[cite: 6]

    max_left_slice_angle = np.ceil(max_left_angle / SLICE_STEP) * SLICE_STEP[cite: 6]
    min_right_slice_angle = np.floor(min_right_angle / SLICE_STEP) * SLICE_STEP[cite: 6]

    # LiDAR 인덱스 계산 (252번이 정면)
    left_index = int(252 + (max_left_slice_angle / SLICE_STEP))[cite: 6]
    right_index = int(252 + (min_right_slice_angle / SLICE_STEP))[cite: 6]
    left_index = max(0, min(503, left_index))[cite: 6]
    right_index = max(0, min(503, right_index))[cite: 6]

    lidar_correction = 0.0[cite: 4]
    
    # LiDAR 배열 내 충돌 위험 감지
    if lidar_ranges is not None and len(lidar_ranges) > 0:[cite: 4]
        ranges = np.array(lidar_ranges)[cite: 4]
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)[cite: 4]

        # 전방(북쪽) 확인
        front_indices = np.concatenate((ranges[230:275], []))[cite: 4]
        min_front = np.min(front_indices) if len(front_indices) > 0 else 10.0[cite: 4]
        
        safe_dist = 0.4 # 40cm[cite: 4]
        
        # 전방에 장애물이 너무 가까울 경우 회피
        if min_front < safe_dist:[cite: 4]
            cam_node_instance.get_logger().warn(f"전방 콘 접근! (거리: {min_front:.2f}m) 회피 기동!")[cite: 4]
            # 좌우 공간 비교 후 넓은 쪽으로 조향
            left_space = np.mean(ranges[350:400])[cite: 4]
            right_space = np.mean(ranges[100:150])[cite: 4]
            
            if left_space > right_space:[cite: 4]
                lidar_correction = -30.0[cite: 4]
            else:[cite: 4]
                lidar_correction = 30.0[cite: 4]
        else:[cite: 4]
            # 측면 거리가 너무 가까우면 중앙으로 밀어내기
            min_left = np.min(ranges[left_index:min(504, left_index+20)]) if left_index < 504 else 10.0
            min_right = np.min(ranges[max(0, right_index-20):right_index]) if right_index > 0 else 10.0
            
            if min_left < 0.25:[cite: 4]
                lidar_correction += 10.0  # 우측으로 밀기[cite: 4]
            if min_right < 0.25:[cite: 4]
                lidar_correction -= 10.0  # 좌측으로 밀기[cite: 4]

    # ---------------------------------------------------------
    # 5. 최종 조향각 산출 및 시각화
    # ---------------------------------------------------------
    calc_angle = steering_angle + lidar_correction[cite: 4]
    
    # 조향각 제한 (-40 ~ 40)
    MAX_ANGLE = 40.0[cite: 4]
    calc_angle = max(min(calc_angle, MAX_ANGLE), -MAX_ANGLE)[cite: 4, 6]

    # 계산된 주행 경로 초록색 진한 선 표시
    draw_len = 250[cite: 6]
    end_x = int(Cx + draw_len * np.sin(np.radians(calc_angle)))[cite: 6]
    end_y = int(Cy - draw_len * np.cos(np.radians(calc_angle)))[cite: 6]
    
    cv2.line(debug_img, (Cx, Cy), (end_x, end_y), (0, 255, 0), 4)[cite: 6]
    cv2.circle(debug_img, (end_x, end_y), 6, (0, 255, 0), -1)[cite: 6]
    
    # 정보 텍스트 표시
    cv2.putText(debug_img, f"Angle: {calc_angle:.1f} deg", (Cx + 20, Cy - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)[cite: 6]
    cv2.putText(debug_img, f"Lidar Idx: {right_index} ~ {left_index}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return calc_angle, debug_img, is_finished[cite: 4, 6]