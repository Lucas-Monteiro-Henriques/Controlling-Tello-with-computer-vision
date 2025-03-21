import cv2
import mediapipe as mp
from djitellopy import Tello
import time
import numpy as np
import pygame

# Inicializa o Tello
drone = Tello()
drone.connect()
print(f'Bateria: {drone.get_battery()}%')

# Inicializa o vídeo do drone
drone.streamon()
time.sleep(2)

# Inicializando MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Dimensões da janela
frame_w, frame_h = 640, 480
cam_center_x, cam_center_y = frame_w // 2, frame_h // 2
ideal_w = 100
dead_zone_side = 5

# Faixas de área para controle de distância
fbRange = [10000, 29000]
pid_up_down = [0.3, 2, 0.005]
pid_yaw = [0.3, 0.5, 0.005]
pid_front_back = [0.2, 1.6, 0.005]
p_x_error = p_y_error = p_z_error = 0
drone.takeoff()
drone.send_rc_control(0, 0, 30, 0)
for_back_velocity = left_right_velocity = up_down_velocity = yaw_velocity = 0

# Inicializa o Pygame
pygame.init()
pygame.display.set_caption("Controle do Tello com MediaPipe")
screen = pygame.display.set_mode([960, 720])

# Função de cálculo de velocidade
def calculate_velocity(error, p_error, pid, limit):
    P, I, D = pid
    velocity = (P * error + D * (error - p_error)) * 0.5
    return int(np.clip(velocity, -1 * limit, limit)), error

# Funções de controle de movimento
def trackFace(info, w, h, p_x_error, p_y_error, p_z_error):
    global for_back_velocity, up_down_velocity, yaw_velocity
    area = info[1]
    x, y = info[0]

    # Cálculo do erro em cada direção
    x_error = x - w // 2
    y_error = y - h // 2
    z_error = 0

    dir_for_back_velocity = 0
    if area != 0:
        print(area)
    if area > fbRange[0] and area < fbRange[1]:
        dir_for_back_velocity = 0
    elif area > fbRange[1]:
        dir_for_back_velocity = -2
    elif area < fbRange[0] and area != 0:
        dir_for_back_velocity = -2

    # Controle de Yaw (Giro) baseado no erro X
    yaw_velocity, p_x_error = calculate_velocity(x_error, p_x_error, pid_yaw, 20)
    # Controle de Altura baseado no erro Y
    up_down_velocity, p_y_error = calculate_velocity(-y_error, p_y_error, pid_up_down, 15)

    # Controle de posição Frontal/Traseira (em relação ao tamanho da face detectada)
    if area != 0:
        z_error = area - (fbRange[0] + fbRange[1]) // 2
        for_back_velocity, p_z_error = calculate_velocity(z_error, p_z_error, pid_front_back, 10)

    # Caso nenhuma face seja detectada
    if x == 0:
        yaw_velocity = 0
        up_down_velocity = 0
        for_back_velocity = 0
        p_x_error = p_y_error = p_z_error = 0

    print(dir_for_back_velocity)
    print(for_back_velocity)
    drone.send_rc_control(0, dir_for_back_velocity * for_back_velocity, up_down_velocity, yaw_velocity)
    return p_x_error, p_y_error, p_z_error


def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if myFaceListArea:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

# Funções de detecção de poses (sem mudanças)
def detect_both_arms_up(pose_landmarks):
    if len(pose_landmarks.landmark) == 33:
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y
    return False

def detect_left_arm_up(pose_landmarks):
    if len(pose_landmarks.landmark) == 33:
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        return left_elbow.y < left_shoulder.y
    return False

def detect_right_arm_up(pose_landmarks):
    if len(pose_landmarks.landmark) == 33:
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return right_elbow.y < right_shoulder.y
    return False

def detect_both_arms_down(pose_landmarks):
    if len(pose_landmarks.landmark) == 33:
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return left_elbow.y > left_shoulder.y and right_elbow.y > right_shoulder.y
    return False


# Loop principal de controle
while True:
    frame_read = drone.get_frame_read()
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Controle de movimentos baseado em poses
        

        if detect_both_arms_down(results.pose_landmarks):
            # Track Tello
            frame, info = findFace(frame)
            p_x_error, p_y_error, p_z_error = trackFace(info, frame_w, frame_h, p_x_error, p_y_error, p_z_error)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            p_x_error, p_y_error, p_z_error = trackFace(info, frame_w, frame_h, p_x_error, p_y_error, p_z_error)

            print("Decolando...")

        elif detect_right_arm_up(results.pose_landmarks):
            drone.send_rc_control(0, 30, 0, 0)  # Avança com velocidade 30
            print("Avançando...")

        elif detect_left_arm_up(results.pose_landmarks):
            drone.send_rc_control(0, -30, 0, 0)  # Recuar com velocidade 30
            print("Recuando...")

        elif detect_arms_crossed(results.pose_landmarks):
            drone.send_rc_control(0, 0 , 10, 0)  # Recuar com velocidade 30
            print("Pousando...")
        else:
            drone.send_rc_control(0, 0, 0, 0)  # Para o movimento
            print("Nenhum")

    # Exibição do vídeo no Pygame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = np.flipud(frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
    pygame.display.update()

   

# Finalizando
cv2.destroyAllWindows()
pose.close()
drone.end()
pygame.quit()
