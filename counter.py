import cv2
import numpy as np
import torch
from mss import mss

# YOLOv8 modelini yükleyin (Ultralytics tarafından sağlanmıştır)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Ekran görüntüsü almak için mss kullanıyoruz
sct = mss()

# Ekranda izlenecek alanı tanımlayın (örnek olarak tüm ekran)
monitor = sct.monitors[1]  # Tüm ekranı kullanmak için

while True:
    # Ekran görüntüsünü al
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # YOLO modelini kullanarak tahmin yap
    results = model(frame)

    # İnsanları tespit etmek için sonuçları filtrele
    person_count = 0
    for det in results.pred[0]:
        # detection format: [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # Sınıf 0 'insan' anlamına gelir
            person_count += 1
            # Tespit edilen insanları dikdörtgen ile çiz
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Ekranda insan sayısını göster
    cv2.putText(frame, f'Insan Sayisi: {person_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Sonucu göster
    cv2.imshow('Konferans Salonu - Insan Sayma', frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cv2.destroyAllWindows()
