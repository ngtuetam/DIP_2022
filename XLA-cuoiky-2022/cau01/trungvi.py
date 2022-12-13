import cv2  
import numpy as np 


# ham loc trung vi
def loc_trung_vi(img):
    '''Ham loc trung vi
    Args:      img: anh goc
    Returns:   img_res: anh ket qua sau khi da qua bo loc trung vi
    '''
    # trich xuat kich thuoc anh
    m, n = img.shape
    # tao mot anh moi de luu ket qua co kich thuoc bang anh goc
    img_res = np.zeros([m, n])
    # bat dau loc trung vi
    for i in range(1, m-1): 
        for j in range(1, n-1): 
            temp = [img[i-1, j-1], 
                   img[i-1, j], 
                   img[i-1, j + 1], 
                   img[i, j-1], 
                   img[i, j], 
                   img[i, j + 1], 
                   img[i + 1, j-1], 
                   img[i + 1, j], 
                   img[i + 1, j + 1]] 
            # thay pixel o giua thanh pixel trung vi (co vi tri la 4)
            temp = sorted(temp) 
            img_res[i, j]= temp[4] 
    # dua cac pixel ve kieu int
    img_res = img_res.astype(np.uint8)
    # tra ve anh ket qua
    return img_res



# doc anh goc
image = cv2.imread('/home/ngtuetam/workspace/XLA-cuoiky-2022/final1.bmp', 0)

# loc trung vi 3x3 va luu anh
img_res = loc_trung_vi(image)
cv2.imwrite('loctrungvi.bmp', img_res) 
