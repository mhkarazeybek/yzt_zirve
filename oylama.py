import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import mediapipe as mp

# YOLOv8x modelini yükleyin (Ultralytics tarafından sağlanmıştır)
model = YOLO('yolov8x.pt')  # YOLOv8x (güncel model olarak değiştirilmiştir)

# Mediapipe yüz algılama modülü
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)

# Ekran görüntüsü almak için mss kullanıyoruz
sct = mss()

# Ekranda izlenecek alanı tanımlayın (örnek olarak tüm ekran)
monitor = sct.monitors[1]  # Tüm ekranı kullanmak için

overtime = 0  # Soru gösterim süresi için sayaç
question_duration = 300  # Belirli bir süre boyunca (örneğin 300 frame) soruyu ekranda tutmak
vote_count = 0  # Oy verenlerin sayısı
total_persons_detected = set()  # Toplamda tespit edilen insanların seti
related_indices = set()  # Ilgili kişiler
unrelated_indices = set()  # Ilgisiz kişiler

while True:
    try:
        # Ekran görüntüsünü al
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # YOLOv8 modelini kullanarak tahmin yap
        results = model(frame, conf=0.2, iou=0.5)  # Daha yüksek güven eşiği kullanarak daha tutarlı sonuçlar elde et
        
        ih, iw, _ = frame.shape  # Frame boyutlarını burada tanımlayın
        person_count = 0
        persons = []
        hands = []

        # İnsanları ve elleri tespit etmek için sonuçları filtrele
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Tespit edilen kutunun koordinatları
                cls = int(box.cls[0])  # Sınıf etiketi
                conf = box.conf[0]  # Güven skoru
                if conf > 0.3:
                    if cls == 0:  # İnsan sınıfı
                        person_count += 1
                        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, iw - 1), min(y2, ih - 1)  # Çerçeve sınırlarını görüntü boyutlarıyla sınırlayın
                        persons.append((x1, y1, x2, y2))
                        # Tespit edilen kişiyi toplam kişilere ekle
                        already_detected = False
                        for person in total_persons_detected:
                            existing_x1, existing_y1, existing_x2, existing_y2 = person
                            iou_x1 = max(existing_x1, x1)
                            iou_y1 = max(existing_y1, y1)
                            iou_x2 = min(existing_x2, x2)
                            iou_y2 = min(existing_y2, y2)
                            iou_width = max(0, iou_x2 - iou_x1)
                            iou_height = max(0, iou_y2 - iou_y1)
                            intersection_area = iou_width * iou_height
                            existing_area = (existing_x2 - existing_x1) * (existing_y2 - existing_y1)
                            current_area = (x2 - x1) * (y2 - y1)
                            iou = intersection_area / float(existing_area + current_area - intersection_area)
                            if iou > 0.05:  # Eğer IoU %5'ten fazlaysa aynı kişi olarak kabul et (daha hassas algılama)
                                already_detected = True
                                break
                        if not already_detected:
                            total_persons_detected.add((x1, y1, x2, y2))
                            unrelated_indices.add(len(total_persons_detected) - 1)
                    elif cls == 15:  # El sınıfı (El tespiti için farklı bir sınıf belirledik, örneğin: 15 (el çantası, el olarak kabul edildi))
                        hands.append((x1, y1, x2, y2))

        # Mediapipe kullanarak yüz tespiti yap ve kişilerin ilgili olup olmadığını belirle
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                # Yüzün merkezi herhangi bir kişi kutusunun içinde mi kontrol et
                for i, (x1, y1, x2, y2) in enumerate(persons):
                    if x1 <= face_center_x <= x2 and y1 <= face_center_y <= y2:
                        if i not in related_indices:
                            related_indices.add(i)
                            if i in unrelated_indices:
                                unrelated_indices.remove(i)

        # El tespiti yap ve oy verenleri belirle
        for hand in hands:
            hand_x1, hand_y1, hand_x2, hand_y2 = hand
            vote_detected = True
            if vote_detected:
                vote_count += 1
                # Elin üzerine yeşil bir nokta ekle
                cx, cy = (hand_x1 + hand_x2) // 2, (hand_y1 + hand_y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Ilgili ve ilgisiz kişileri belirlemek için işaretleme yap
        for i, (x1, y1, x2, y2) in enumerate(persons):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if i in related_indices:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Ilgili - Yeşil nokta
            elif i in unrelated_indices:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Ilgisiz - Kırmızı nokta

        # Ekranda insan sayısını, ilgili/ilgisiz ve oy verenlerin sayısını göster
        cv2.putText(frame, f'Anlik Insan Sayisi: {person_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Toplam Insan Sayisi: {len(total_persons_detected)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f'Oy Verenler: {vote_count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Sonucu göster
        resized_frame = cv2.resize(frame, (960, 540))
        cv2.imshow('Konferans Salonu - Insan Sayma ve Ilgi Analizi', resized_frame)

        # 'q' tuşuna basarak çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        break

# Kaynakları serbest bırak
cv2.destroyAllWindows()
