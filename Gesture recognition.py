import cv2
import mediapipe as mp
import math
from pythonosc.udp_client import SimpleUDPClient

# OSC 设置（发送到 TouchDesigner）
osc_ip = "127.0.0.1"
osc_port = 8080
client = SimpleUDPClient(osc_ip, osc_port)

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

def get_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# 状态控制变量
thumb_index_visible = False
last_thumb_index_touch = False

cross_hand_visible = False
last_cross_touch = False

# 参数设置
thumb_index_threshold = 0.05
cross_touch_threshold = 0.08
paired_fingertips = [(4, 4), (8, 8), (12, 12), (16, 16), (20, 20)]

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    left_hand = None
    right_hand = None

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            hand_label = label.lower()
            if label == "Left":
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks

            # ---------- 单手拇指-食指触摸开关 ----------
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            dist = get_distance(thumb_tip, index_tip)
            is_thumb_index_touch = dist < thumb_index_threshold

            print(f"[SEND {label.upper()}] Pinch: {dist:.3f}, "
                  f"Thumb: [{thumb_tip.x:.3f}, {thumb_tip.y:.3f}], "
                  f"Index: [{index_tip.x:.3f}, {index_tip.y:.3f}]")

            if is_thumb_index_touch and not last_thumb_index_touch:
                thumb_index_visible = not thumb_index_visible

            last_thumb_index_touch = is_thumb_index_touch

            if thumb_index_visible:
                x1, y1 = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                x2, y2 = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

                # 分别发送 OSC 数值
                client.send_message(f"/{hand_label}/thumb/x", thumb_tip.x)
                client.send_message(f"/{hand_label}/thumb/y", thumb_tip.y)
                client.send_message(f"/{hand_label}/index/x", index_tip.x)
                client.send_message(f"/{hand_label}/index/y", index_tip.y)

            # 绘制关键点
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ---------- 双手对指合掌触发 ----------
        if left_hand and right_hand:
            close_count = 0
            for idx_l, idx_r in paired_fingertips:
                dist = get_distance(left_hand.landmark[idx_l], right_hand.landmark[idx_r])
                if dist < cross_touch_threshold:
                    close_count += 1

            is_cross_touch = close_count >= 3

            if is_cross_touch and not last_cross_touch:
                cross_hand_visible = not cross_hand_visible

            last_cross_touch = is_cross_touch

            if cross_hand_visible:
                for idx_l, idx_r in paired_fingertips:
                    point_l = left_hand.landmark[idx_l]
                    point_r = right_hand.landmark[idx_r]

                    x1, y1 = int(point_l.x * frame.shape[1]), int(point_l.y * frame.shape[0])
                    x2, y2 = int(point_r.x * frame.shape[1]), int(point_r.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # 分别发送四个点的值
                    client.send_message(f"/cross/line_{idx_l}/x1", point_l.x)
                    client.send_message(f"/cross/line_{idx_l}/y1", point_l.y)
                    client.send_message(f"/cross/line_{idx_l}/x2", point_r.x)
                    client.send_message(f"/cross/line_{idx_l}/y2", point_r.y)

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
