import cv2
import numpy as np
import torch
from mss import mss
import random

# YOLOv8 modelini yükleyin (Ultralytics tarafından sağlanmıştır)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu', force_reload=True)  # CUDA sorunlarını önlemek için CPU kullanılıyor, force_reload ekledik

# Ekran görüntüsü almak için mss kullanıyoruz
sct = mss()

# Ekranda izlenecek alanı tanımlayın (örnek olarak tüm ekran)
monitor = sct.monitors[1]  # Tüm ekranı kullanmak için

# Duygu analizi ve el tespiti için Google Mediapipe kullanımı
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Mediapipe yüz ağız çözümü ve el çözümü
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

overtime = 0  # Soru gösterim süresi için sayaç
question_duration = 300  # Belirli bir süre boyunca (örneğin 300 frame) soruyu ekranda tutmak
vote_count = 0  # Oy verenlerin sayısı

while True:
    try:
        # Ekran görüntüsünü al
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # YOLO modelini kullanarak tahmin yap
        with torch.no_grad():  # Performans sorunlarını ve uyarıları önlemek için no_grad kullanıyoruz
            results = model(frame)

        # İnsanları tespit etmek için sonuçları filtrele
        ih, iw, _ = frame.shape  # Frame boyutlarını burada tanımlayın
        person_count = 0
        persons = []
        for det in results.xyxy[0]:
            # detection format: [x1, y1, x2, y2, confidence, class]
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # Sınıf 0 'insan' anlamına gelir
                person_count += 1
                persons.append((int(x1), int(y1), int(x2), int(y2)))

        # Duygu analizi için yüz tespiti
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        hands_results = hands.process(rgb_frame)
        related_indices = set()
        vote_count = 0

        face_y_coords = []  # Her bir yüz için y koordinatlarını saklamak için liste

        if face_results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(face_results.multi_face_landmarks):
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                # Yüzün hangi kişiye ait olduğunu belirlemek için yüzün merkezini hesaplayın
                x_coords = [landmark.x * iw for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * ih for landmark in face_landmarks.landmark]
                face_x = int(sum(x_coords) / len(x_coords))
                face_y = int(sum(y_coords) / len(y_coords))
                face_y_coords.append(face_y)  # Yüz y koordinatını sakla
                
                # Yüzün ilgili olup olmadığını belirtmek için en yakın kişiyle eşleştirin
                for i, (x1, y1, x2, y2) in enumerate(persons):
                    if x1 <= face_x <= x2 and y1 <= face_y <= y2:
                        # El tespit et ve kişinin oy verip vermediğini kontrol et
                        if hands_results.multi_hand_landmarks:
                            for hand_landmarks in hands_results.multi_hand_landmarks:
                                # El bileği ve parmak uçlarını kontrol et
                                hand_points = [
                                    mp_hands.HandLandmark.WRIST,
                                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                    mp_hands.HandLandmark.RING_FINGER_TIP,
                                    mp_hands.HandLandmark.PINKY_TIP
                                ]
                                vote_detected = False
                                wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * iw)
                                wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * ih)
                                # Elin yüz hizasından yukarıda olup olmadığını kontrol et
                                if wrist_y < face_y - (0.2 * (y2 - y1)):  # Yüz hizasından belirgin yukarıda olduğunda oy kullanılmış kabul edilecek
                                    vote_detected = True
                                if vote_detected and i not in related_indices:
                                    # El kişinin kutusu içerisindeyse ve yüz hizasından yukarıdaysa oy verilmiş olarak kabul et
                                    # Elin üzerine yeşil bir nokta ekle
                                    for point in hand_points:
                                        hand_x = int(hand_landmarks.landmark[point].x * iw)
                                        hand_y = int(hand_landmarks.landmark[point].y * ih)
                                        cv2.circle(frame, (hand_x, hand_y), 5, (0, 255, 0), -1)
                                    cv2.rectangle(frame, (x1, y1, x2, y2), (0, 255, 0), 2)
                                    # Oy verenlerin boxlarının üzerinde yeşil nokta ekle
                                    cv2.circle(frame, (x1 + 10, y1 + 10), 5, (0, 255, 0), -1)  # Ilgili - Yesil çerçeve
                                    related_indices.add(i)
                                    vote_count += 1  # Oylama olarak kabul edilen her ilgili kişi
                                    break
                        else:
                            # Eğer el tespit edilmediyse normal olarak işaretle
                            cv2.rectangle(frame, (x1, y1, x2, y2), (0, 255, 0), 2)
                        break

        # Tespit edilen tüm insanlar (ilgili ve ilgisiz) için oy kullanma tespiti
        if hands_results.multi_hand_landmarks:
            for i, (x1, y1, x2, y2) in enumerate(persons):
                if i not in related_indices:  # İlgili olarak işaretlenmemiş olanlar için oy tespiti
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        # El bileği ve parmak uçlarını kontrol et
                        hand_points = [
                            mp_hands.HandLandmark.WRIST,
                            mp_hands.HandLandmark.INDEX_FINGER_TIP,
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                            mp_hands.HandLandmark.RING_FINGER_TIP,
                            mp_hands.HandLandmark.PINKY_TIP
                        ]
                        vote_detected = False
                        wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * iw)
                        wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * ih)
                        # Yüz y koordinatını al
                        face_y = face_y_coords[i] if i < len(face_y_coords) else y2  # Eğer yüz bulunamadıysa yüz hizası olarak kişinin alt sınırını kabul et
                        # Elin kutu içinde olup olmadığını ve yüz hizasından yukarıda olup olmadığını kontrol et
                        if x1 <= wrist_x <= x2 and wrist_y < face_y - (0.2 * (y2 - y1)):
                            vote_detected = True
                        if vote_detected:
                            # El kişinin kutusu içerisindeyse ve yüz hizasından yukarıdaysa oy verilmiş olarak kabul et
                            # Elin üzerine yeşil bir nokta ekle
                            for point in hand_points:
                                hand_x = int(hand_landmarks.landmark[point].x * iw)
                                hand_y = int(hand_landmarks.landmark[point].y * ih)
                                cv2.circle(frame, (hand_x, hand_y), 5, (0, 255, 0), -1)
                            cv2.rectangle(frame, (x1, y1, x2, y2), (0, 0, 255), 2)  # Ilgisiz ama oy veren - Kırmızı çerçeve
                            cv2.circle(frame, (x1 + 10, y1 + 10), 5, (0, 255, 0), -1)  # Oy veren - Yesil nokta
                            vote_count += 1  # Oylama olarak kabul edilen kişi
                            break
        
        # Tespit edilen diğer insanların çerçevesini kırmızı olarak çiz
        unrelated_count = 0
        for i, (x1, y1, x2, y2) in enumerate(persons):
            if i in related_indices:
                continue  # İlgili olanlar zaten çizildi
            else:
                cv2.rectangle(frame, (x1, y1, x2, y2), (0, 0, 255), 2)  # Ilgisiz - Kırmızı çerçeve
                unrelated_count += 1
        
        # Ekranda insan sayısını ve ilgili/ilgisiz sayısını göster
        related_count = len(related_indices)
        cv2.putText(frame, f'Insan Sayisi: {person_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Ilgili: {related_count}  Ilgisiz: {unrelated_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'Oy Verenler: {vote_count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Sonucu göster
        cv2.imshow('Konferans Salonu - Insan Sayma ve Ilgi Analizi', frame)

        # 'q' tuşuna basarak çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        break

# Kaynakları serbest bırak
cv2.destroyAllWindows()
