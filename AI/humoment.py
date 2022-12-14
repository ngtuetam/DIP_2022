import cv2
import numpy as np
import matplotlib as plt
import glob


humoment_data = []
holder = []
img_number = 1
while img_number < 21:
    path = "/home/ngtuetam/workspace/Crab Dataset/"+str(img_number)+".jpg"
    im = cv2.imread(path)
    im = im/255
    im = np.round_(im)
    X,Y,H = im.shape
    sum = 0
    # tinh dien tich cua vung anh mau trang
    for i in range(X):
        for j in range(Y):
            if (im[i][j][0] == 1):
                sum = sum + 1
    m00 = sum 
    sumX = 0
    sumY = 0
    # Vi tri trung tam cua vung anh mau trang : x_bar, y_bar
    for i in range(X):
        for j in range(Y):
            sumX = sumX + i*im[i][j][0]
            sumY = sumY + j*im[i][j][0]
    x_bar = sumX/m00
    x_bar = np.round_(x_bar,2)
    y_bar = sumY/m00
    y_bar = np.round_(y_bar,2)

    # Tính các moment trung tâm m02, m20, m11, m30, m03, m12, m21
    def calculateMoment(p,q):
        m_pq = 0
        for i in range(X):
            for j in range(Y):
                m_pq = m_pq + ((i - x_bar)**p)*((j-y_bar)**q)*im[i][j][0]
        return m_pq

    m02 =  calculateMoment(0,2)
    m20 =  calculateMoment(2,0)
    m11 =  calculateMoment(1,1)
    m30 =  calculateMoment(3,0)
    m03 =  calculateMoment(0,3)
    m12 =  calculateMoment(1,2)
    m21 =  calculateMoment(2,1)


    # Tính các moment trung tâm chuẩn hóa M02,M20, M11, M03, M30, M12, M21
    def calculateMomentStand(p,q,mpq,m00):
        M_pq = (mpq)/(m00 ** (((p+q)/2)+1))
        return M_pq

    M02 = calculateMomentStand(0,2,m02,m00)
    M20 = calculateMomentStand(2,0,m20,m00)
    M11 = calculateMomentStand(1,1,m11,m00)
    M30 = calculateMomentStand(3,0,m30,m00)
    M03 = calculateMomentStand(0,3,m03,m00)
    M12 = calculateMomentStand(1,2,m12,m00)
    M21 = calculateMomentStand(2,1,m21,m00)

    # Tính các moment HU 
    S1 = M20 + M02
    S2 = (M20 - M02)*(M20+M02)+4*M11*M11
    S3 = (M30 -3*M12)*(M30 -3*M12)+ (M30 -3*M21)*(M30 -3*M21)
    S4 = (M30 +M12)*(M30 +M12)+ (M03 +M21)*(M03 +M21)
    S5 = (M30-3*M12)*(M30+M12)*((M30+M12)*(M30+M12)-3*(M03+M21)*(M03+M21)) + (3*M12-M03)*(M03+M21)*(3*(M30+M12)*(M30+M12)-(M03+M21)*(M03+M21))
    S6 = (M20-M02)*((M30+M12)*(M30+M12)-(M03+M21)*(M03+M21)) +4*M11*(M30+M12)*(M03+M21)
    S7 = (3*M21-M03)*(M30+M12)*((M30+M12)*(M30+M12)-3*(M03+M21)*(M03+M21)) + (M30-3*M12)*(M21+M02)*(3*(M30+M12)*(M30+M12)-(M03+M21)*(M03+M21))
    
    # phuong phap holdout
    if (img_number <= 10):
        holder = [np.round(S1,5), np.round(S2,5), np.round(S3,5), np.round(S4,5), np.round(S5,5), np.round(S6,5), np.round(S7,5)]
        humoment_data.append(holder)
    if (img_number >= 11):
        holder = [np.round(S1,5), np.round(S2,5), np.round(S3,5), np.round(S4,5), np.round(S5,5), np.round(S6,5), np.round(S7,5)]
        humoment_data.append(holder)
    img_number = img_number + 1