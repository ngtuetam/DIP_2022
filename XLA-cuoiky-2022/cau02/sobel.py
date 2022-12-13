import cv2
import matplotlib.pyplot as plt
import numpy as np

def ky_thuat_sobel(img):
    '''Phat hien duong bang ky thuat Sobel
    Args:     img : anh goc
    Returns:  img_res: anh ket qua sau khi phat hien duong bang ky thuat Sobel
    '''

    sobel_x = np.array(
        [[1.0, 0.0, -1.0], 
         [2.0, 0.0, -2.0], 
         [1.0, 0.0, -1.0]])
    sobel_y = np.array(
        [[1.0, 2.0, 1.0], 
         [0.0, 0.0, 0.0], 
         [-1.0, -2.0, -1.0]])

    # trich xuat thong tin kich thuoc anh
    m, n = img.shape

    # tao mot anh moi de luu ket qua co kich thuoc bang anh goc
    img_res = np.zeros([m, n])

    for i in range(m - 2):
        for j in range(n - 2):
            gx = np.sum(np.multiply(sobel_x, img[i:i + 3, j:j + 3]))
            gy = np.sum(np.multiply(sobel_y, img[i:i + 3, j:j + 3]))
            img_res[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)

    return img_res


# doc anh goc
image = cv2.imread('/home/ngtuetam/workspace/XLA-cuoiky-2022/final2.jpg', cv2.IMREAD_GRAYSCALE)

# phat hien canh bang ky thuat sobel 
img_res = ky_thuat_sobel(image)
plt.imsave('anh_sobel.jpg', img_res, cmap=plt.get_cmap('gray'))