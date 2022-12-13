import numpy as np
import matplotlib.pyplot as plt
import cv2
 

def k_means(image, K):
    '''Thuat toan K-Means
    Args:       image: hinh anh goc
                K: so cum
    Returns:    img_res : anh ket qua da duoc phan doan mau sac
    '''

    # vi cv2 doc anh theo thu tu BGR nen ta chuyen ve RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # reshape hinh anh thanh 2 chieu co 3 gia tri RGB
    img_vectorized = image.reshape((-1,3))   #(360000x3)
    # chuyen ve kieu float
    img_vectorized = np.float32(img_vectorized)

    #kmeans dung lai khi dat 100 vong lap hoac do chinh xac = 85% (epsilon = 0.85)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    

    # chon khoi tao centers theo kieu ngau nhien (random) cho k-means 
    retval, labels, centers = cv2.kmeans(img_vectorized, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert center ve gia tri 8-bit 
    centers = np.uint8(centers)
    
    # áp dụng các giá trị trọng tâm (cũng là R,G,B) cho tất cả các pixel, 
    # sao cho hình ảnh thu được sẽ có số lượng màu được chỉ định.
    result_data = centers[labels.flatten()]
    
    # reshape data into the original image dimensions
    img_res = result_data.reshape((image.shape))
    return img_res

# doc anh goc
image = cv2.imread('/home/ngtuetam/workspace/XLA-cuoiky-2022/cau03/final3.jpg')
# ap dung k-means
img_res = k_means(image,10)
# luu anh ket qua 
plt.imsave('kmeans_10.jpg', img_res, cmap=plt.get_cmap('gray'))
