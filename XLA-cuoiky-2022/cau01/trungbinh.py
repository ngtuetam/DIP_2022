import cv2  
import numpy as np 
# import matplotlib.pyplot as plt 


# ham loc trung binh
def loc_trung_binh(img, mask):
    ''' Ham loc trung binh
    Args:          img: anh goc
                   mask: mat na loc trung binh
    Return:        img_res: anh da duoc chap voi bo loc
    '''
    # trich xuat kich thuoc anh
    m, n = img.shape
    # tao mot anh moi de luu ket qua co kich thuoc bang anh goc
    img_res = np.zeros([m, n])
    # chap anh voi bo loc
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp =  img[i - 1, j - 1]    * mask[0, 0] \
                  + img[i - 1, j]        * mask[0, 1] \
                  + img[i - 1, j + 1]    * mask[0, 2] \
                  + img[i, j - 1]        * mask[1, 0] \
                  + img[i, j]            * mask[1, 1] \
                  + img[i, j + 1]        * mask[1, 2] \
                  + img[i + 1, j - 1]    * mask[2, 0] \
                  + img[i + 1, j]        * mask[2, 1] \
                 + img[i + 1, j + 1]     * mask[2, 2]
            img_res[i, j] = temp

    # dua cac pixel ve kieu int
    img_res = img_res.astype(np.uint8)
    # tra ve anh ket qua
    return img_res

# tao mot ma tran loc trung binh 3x3
mask3x3 = np.ones((3, 3), dtype="float") * (1.0 / (3 * 3))


# doc anh goc
image = cv2.imread('/home/ngtuetam/workspace/XLA-cuoiky-2022/final1.bmp', 0)

# loc trung binh anh va luu ket qua
image_res = loc_trung_binh(image, mask3x3)
cv2.imwrite('loctrungbinh.bmp', image_res)
