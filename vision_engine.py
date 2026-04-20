import cv2
import numpy as np
from ultralytics import YOLO

model=YOLO('yolov8n.pt')

def get_analysed_results(img):
    results=model(img,conf=0.15)
    shine,redness,shape=0,0,0
    dog_detected=False
    final_img=img.copy()

    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])]=='dog':
                dog_detected=True
                x1,y1,x2,y2=map(int,box.xyxy[0])
                dog_crop=img[max(0,y1):min(img.shape[0],y2),max(0,x1):min(img.shape[0],x2)]
                shine=calculate_shine(dog_crop)
                redness=check_redness(dog_crop)
                shape=check_body_shape(dog_crop)

                final_img=check_diagnostics(img,(x1,y1,x2,y2),shine,redness,shape)
                break

    return final_img,shine,redness,shape,dog_detected

def calculate_shine(roi):
    imggray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    avg_v=hsv[:,:,2].mean()

    variance=cv2.Laplacian(imggray,cv2.CV_64F).var()

    base_score=40

    log_variance=np.log1p(variance)

    if avg_v>160:
        divisor=5.3
    
    else:
        divisor=5.8
    
    score= (log_variance/divisor) * 100

    final_score=int(min(max(base_score,score),100))
    return final_score

def check_redness(roi):
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    lower_red1=np.array([0,50,50])
    upper_red1=np.array([12,255,255])

    lower_red2=np.array([170,50,50])
    upper_red2=np.array([180,255,255])

    mask1=cv2.inRange(hsv,lower_red1,upper_red1)
    mask2=cv2.inRange(hsv,lower_red2,upper_red2)

    full_red_mask=cv2.bitwise_or(mask1,mask2)

    red_pixel=np.sum(full_red_mask>0)
    total_pixel=full_red_mask.size

    percentage= (red_pixel/total_pixel) * 100

    return round(percentage,2)

def check_body_shape(roi):
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(7,7),0)
    canny=cv2.Canny(blur,50,150)
    edge_pixels=np.count_nonzero(canny)
    total_pixels=canny.size

    percentage=(edge_pixels/total_pixels)*100
    return round(percentage,2)

def check_diagnostics(img,box,shine,redness,shape):
    output=img.copy()
    x1,y1,x2,y2=box

    cv2.rectangle(output,(x1,y1),(x2,y2),(0,255,0),3)
    
    cv2.rectangle(output,(0,0),(img.shape[1],60),(0,0,0),-1)

    font=cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(output,f"SHINE : {shine}% ",(20,40),font,0.7,(255,255,255),2)
    cv2.putText(output,f"SKIN : {redness}% ",(int(img.shape[1]/2.5),40),font,0.7,(0,0,255),2)
    cv2.putText(output,f"SHAPE : {shape}% ",(int(img.shape[1]/1.4),40),font,0.7,(0,255,0),2)

    return output



               
