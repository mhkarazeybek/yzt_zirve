import cv2
import numpy as np
import torch
from mss import mss

# YOLOv8 modelini yükleyin (Ultralytics tarafından sağlanmıştır)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu', force_reload=True)  # CUDA sorunlarını önlemek için CPU kullanılıyor, force_reload ekledik

# Ekran görüntüsü almak için mss kullanıyoruz
sct = mss()

# Ekranda izlenecek alanı tanımlayın (örnek olarak tüm ekran)
monitor = sct.monitors[1]  # Tüm ekranı kullanmak için

# Duygu analizi için Google Mediapipe kullanımı
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Mediapipe yüz ağız çözümü
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

while True:
    try:
        # Ekran görüntüsünü al
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # YOLO modelini kullanarak tahmin yap
        results = model(frame)

        # İnsanları tespit etmek için sonuçları filtrele
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
        related_indices = set()
        if face_results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(face_results.multi_face_landmarks):
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                # Yüzün hangi kişiye ait olduğunu belirlemek için yüzün merkezini hesaplayın
                ih, iw, _ = frame.shape
                x_coords = [landmark.x * iw for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * ih for landmark in face_landmarks.landmark]
                face_x = int(sum(x_coords) / len(x_coords))
                face_y = int(sum(y_coords) / len(y_coords))
                
                # Yüzün ilgili olup olmadığını belirtmek için en yakın kişiyle eşleştirin
                for i, (x1, y1, x2, y2) in enumerate(persons):
                    if x1 <= face_x <= x2 and y1 <= face_y <= y2:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Ilgili - Yesil çerçeve
                        related_indices.add(i)
                        break
        
        # Tespit edilen diğer insanların çerçevesini kırmızı olarak çiz
        unrelated_count = 0
        for i, (x1, y1, x2, y2) in enumerate(persons):
            if i in related_indices:
                continue  # İlgili olanlar zaten çizildi
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Ilgisiz - Kırmızı çerçeve
                unrelated_count += 1
        
        # Ekranda insan sayısını ve ilgili/ilgisiz sayısını göster
        related_count = len(related_indices)
        cv2.putText(frame, f'Insan Sayisi: {person_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Ilgili: {related_count}  Ilgisiz: {unrelated_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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