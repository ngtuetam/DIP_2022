import cv2 
import numpy as np
import matplotlib.pyplot as plt
  
def ky_thuat_roberts(img):
    '''Phat hien duong bang ky thuat Roberts
    Args:     img : anh goc
    Returns:  img_res: anh ket qua sau khi phat hien duong bang ky thuat Roberts
    '''
    roberts_vertical = np.array( [[1, 0 ],
                                  [0,-1 ]] )
    roberts_horizontal = np.array([[ 0, 1 ],
                                   [-1, 0 ]])
    
    # trich xuat thong tin kich thuoc anh
    m, n = img.shape

    # tao mot anh moi de luu ket qua co kich thuoc bang anh goc
    img_res = np.zeros([m, n])

    for i in range(m - 1):
        for j in range(n - 1):
            gx = np.sum(np.multiply(roberts_vertical, img[i:i + 2, j:j + 2]))
            gy = np.sum(np.multiply(roberts_horizontal, img[i:i + 2, j:j + 2]))
            img_res[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
    return img_res

# doc anh goc
image = cv2.imread('/home/ngtuetam/workspace/XLA-cuoiky-2022/final2.jpg', cv2.IMREAD_GRAYSCALE)

# phat hien canh bang ky thuat Roberts
img_res = ky_thuat_roberts(image)
plt.imsave('anh_roberts.jpg', img_res, cmap=plt.get_cmap('gray'))