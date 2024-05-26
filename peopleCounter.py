
import cv2
import time
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Burada modeli yüklüyoruz. Ben x sürümünü kullandım. En yavaş çalışan sürüm bu
# Ama en iyi sonuçları veren sürüm de bu. Eğer daha hızlı olsun derseniz diğer
# düşük sürümleri kullanabilirsiniz.
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Kullanılacak videoyu tanımladık. 
kamera = cv2.VideoCapture("video.mp4")
font = cv2.FONT_HERSHEY_DUPLEX

# FPS hesaplamak için kullanacağız bunları. 
prev_frame_time = 0
new_frame_time = 0

# Burada çizginin ve bölgenin koordinatları videoya göre değişir. Benim videonun boyutu 640*360 olduğu için ve orta hizaya 
# gelmesini istediğim için çizginin kordinatları 0,130 ve 640,280
# O yüzden bölge kordinatları da Bu çizginin 20 piksel yukarısında ve aşağısında olacak iki tane paralel kenara uygun olacak 
# şekilde

# yukarıdaki bölge
region1=np.array([(0,130),(0,110),(640,260),(640,280)])
region1 = region1.reshape((-1,1,2))

# aşağı bölge 
region2=np.array([(0,130),(0,150),(640,300),(640,280)])
region2 = region2.reshape((-1,1,2))

# Üst bölgeye giren kişilerin id'leri burada tutuluyor
total_ust=set()

# Alt bölgeye giren kişilerin id'leri burada tutuluyor
total_alt=set()


# Gelen(Giren) kişileri id'leri burada tutuluyor
first_in=set()

# Giden (Çıkan) kişileri id'leri burada tutuluyor
first_out=set()

while True:

    # FPS'i hesaplaıyoruz. İsterseniz ekrana yazdırabilirsiniz
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    #print(fps)
    
    
    
    ret, frame = kamera.read()
    

    
    if not ret:
        break
    
    # Resmi RGB uzayına dönüştürdük    
    rgb_img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # Çizgiyi ekrana çizdiriyoruz. 
    cv2.line(frame, (0,130), (640,280), (255,0,255), 3)
    # Resmi modele veriyoruz. Tracking yapılacağı için track modunda verdik.
    results = model.track(rgb_img, persist=True, verbose=False)
    # Kaç tane sonuç bulunmuşsa o kadar dönece bu döngü
    for i in range(len(results[0].boxes)):
        # Burada her bir nesne için döngü dönecek. Burada yapılan işlemlere her bir nesneye uygulanıyor o yüzden
        # Tespit edilen nesnelerin konumları.
        # x1 ve y1 sol üst köşe. x2 ve y2 sağ alt köşe koordinatları.
        x1,y1,x2,y2=results[0].boxes.xyxy[i]
        # Tespit edilen nesnelerin score değerleri.
        score=results[0].boxes.conf[i]
        # Tespit edilen nesnelerin ait hangi sınıfa ait oldukları
        cls=results[0].boxes.cls[i]
        # Tespit edilen nesnelerin id'leri 
        ids=results[0].boxes.id[i]
        
        # Değerleri uygun bir fomata çevirdik.
        x1,y1,x2,y2,score,cls,ids=int(x1),int(y1),int(x2),int(y2),float(score),int(cls),int(ids)
        
        # burada 0.5' lik bir threshold uyguladık. Tespit edilme değeri daha küçükse değerlendirmeye almıyoruz.
        if score<0.5:
            continue
        # İnsan sayacağımız için insan dışındaki diğer nesneleri değerlendirmeye almıyoruz.
        if cls!=0:
            continue

        # İsterseniz nesneleri kutuna içine alabilirsiniz aşağıdaki komut ile
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        
        # Nesnelerin ortasının konumu buluyoruz.
        cx=int(x1/2+x2/2)
        cy=int(y1/2+y2/2)
        # nesnelerin orta noktalrını bir daire olarak ekranda göstereceğiz.
        cv2.circle(frame,(cx,cy),4,(0,255,255),-1)
        
        # Nesnenin orta noktasının üst bölgeye girip girmedğine bakıyoruz
        inside_region1=cv2.pointPolygonTest(region1,(cx,cy),False)
        
        # Eğer nesnenin orta noktası üst bölgeye girmişse if bloğunun içine girilir.
        if inside_region1>0:
            
            
            # Bu komut ile nesnenin alt bölgeye girip girmediğne bakıyoruz. 
            # Bu bloğun içinde olduğumuz için zaten üst bölgesinin içinde bulunuyor. 
            if ids in total_alt:
                # O yüzden eğer alt bölgeye girilmişse kişi daha öncesinde girmiştir.
                # Bu durumda kişi alt sonra üst bölgeye girmiştir. Bu da kişinin kameradan uzaklaştığını gösterir.
                # Bu yüzden kişi çıkıyordur.
                # Çıkan kişi listesine bu kişinin id'sini ekliyoruz.
                first_out.add(ids)
                #Çizginin rengini  değiştridik. Kişi çıktığı için 
                cv2.line(frame, (0,130), (640,280), (255,255,255), 3)
            # Üst bölgeye giren kişinin id'sini ekledik.
            # Bunu alt bölgeye girip girilmediğini kontrol ettikten sonra yapıyoruz. 
            total_ust.add(ids)
            
        # Eğer nesnenin orta noktası alt bölgeye girmişse if bloğunun içine girilir.
        # Yukarıdaki ile benzer bir mantık var
        inside_region2=cv2.pointPolygonTest(region2,(cx,cy),False)
        if inside_region2>0:
            if ids in total_ust:
                cv2.line(frame, (0,130), (640,+280), (255,255,255), 3)
                first_in.add(ids)
            total_alt.add(ids)
        
    
    
    # Giren ve çıkan kişi sayısını ekranda göstermek için bunları string formatına çevirdik
    # Burada giren ve çıkan kişi sayısı bunlarda kaç tane eleman varsa o kadardır.
    # Burada set kullanma sebebimiz, bunlarda aynı elemanın sadece bir kere bulunmasıyla ilgili.
    # Eğer liste olsaydı aynı eleman birden fazla olabilirdi
    first_in_str='IN: '+str(len(first_in))
    first_out_str='OUT: '+str(len(first_out))
    
    # Ekrana yazdırırken güzel dursun diye görüntünün sol üst ve sağ üst köşesine arka plan rengi ayarladık.
    frame[0:40,0:120]=(102,0,153)
    frame[0:40,510:640]=(102,0,153)
    # Ekranda gösteriyoruz.
    cv2.putText(frame, first_in_str,(0, 30), font, 1, (255,255,255), 1)
    cv2.putText(frame, first_out_str,(510, 30), font, 1, (255,255,255), 1)
    #print('IN: ', len(first_in), 'OUT: ', len(first_out))
    
    # İsterseniz bölgeleri ekranda gösterebilrsiniz.
    #cv2.polylines(frame,[region1],True,(255,0,0),2)
    #cv2.polylines(frame,[region2],True,(255,0,255),2)
    
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()