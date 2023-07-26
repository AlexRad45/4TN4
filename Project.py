import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

def BGR_to_YUV1(bgr):
    # Convert BGR to YUV
    b, g, r = cv2.split(bgr)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b
    yuv_image = cv2.merge([y, u, v])
    return yuv_image

def YUV_to_BGR1(yuv):
    # Convert YUV to BGR
    y, u, v = cv2.split(yuv)
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    bgr_image = cv2.merge([b, g, r])
    return np.uint8(bgr_image)


def BGR_to_YUV(bgr):
    # Conversion matrix for BGR to YUV color space
    conv_mat = np.array([[0.114, 0.587, 0.299],
                         [0.436, -0.289, -0.147],
                         [-0.100, -0.515, 0.615]])

    # Convert the image to YUV color space
    h, w, c = bgr.shape
    bgr = bgr.astype("float32")
    img_yuv = np.zeros((h, w, c), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            img_yuv[i, j, :] = np.dot(conv_mat, bgr[i, j, :])

    # Clip pixel values to range 0-255
    #img_yuv = np.clip(img_yuv, 0, 255)

    return img_yuv

def YUV_to_BGR(yuv):
    # Conversion matrix for YUV to BGR color space
    conv_mat = np.array([[1.0, 2.032, 0],
                         [1.0, -0.395, -0.581],
                         [1.0, 0, 1.140]])

    # Convert the image back to BGR color space
    h, w, c = yuv.shape
    img_bgr = np.zeros((h, w, c), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            img_bgr[i, j, :] = np.dot(conv_mat, yuv[i, j, :])

    # Clip pixel values to range 0-255
    #img_bgr = np.clip(img_bgr, 0, 255)

    return np.uint8(img_bgr)


def downsample(img):
    r,c,d = img.shape
    y_img = np.zeros((r//2,c//2),dtype=np.uint8)
    u_img = np.zeros((r//4,c//4),dtype=np.uint8)
    v_img = np.zeros((r//4,c//4),dtype=np.uint8)
    
    y, u, v = cv2.split(img)
    
    for i in range(0,r//2):
        for j in range(0,c//2):
            y_img[i,j] = y[i*2,j*2]
    
    for k in range(0,r//4):
        for l in range(0,c//4):
            u_img[k,l] = u[k*4,l*4]
            v_img[k,l] = v[k*4,l*4]
    
    return y_img, u_img, v_img
    
def upsample(y,u,v):
    row,col = y.shape
    y_img = np.zeros((row*2,col*2),dtype=np.uint8)
    u_img = np.zeros((row*2,col*2),dtype=np.uint8)
    v_img = np.zeros((row*2,col*2),dtype=np.uint8)
    
    for i in range(int(row*2)):
        for j in range(int(col*2)):
            i_old_y = i//2
            j_old_y = j//2
            
            i_old_uv = i//4
            j_old_uv = j//4
            
            a = (i/2)-i_old_y
            b = (j/2)-j_old_y
            c = (i/4)-i_old_uv
            d = (j/4)-j_old_uv
            
            if((i_old_y < 199) and (j_old_y < 299)):
                y_img[i,j] = (1-a)*(1-b)*y[i_old_y,j_old_y] + a*(1-b)*y[i_old_y+1,j_old_y] + (1-a)*b*y[i_old_y,j_old_y+1] + a*b*y[i_old_y+1,j_old_y+1]
            if((i_old_y == 199) and (j_old_y == 299)):
                y_img[i,j] = y[199,299]
                
            if((i_old_uv < 99) and (j_old_uv < 149)):
                u_img[i,j] = (1-c)*(1-d)*u[i_old_uv,j_old_uv] + c*(1-d)*u[i_old_uv+1,j_old_uv] + (1-c)*d*u[i_old_uv,j_old_uv+1] + c*d*u[i_old_uv+1,j_old_uv+1]
                v_img[i,j] = (1-c)*(1-d)*v[i_old_uv,j_old_uv] + c*(1-d)*v[i_old_uv+1,j_old_uv] + (1-c)*d*v[i_old_uv,j_old_uv+1] + c*d*v[i_old_uv+1,j_old_uv+1]
            if((i_old_uv == 99) and (j_old_uv == 149)):
                u_img[i,j] = u[99,149]
                v_img[i,j] = v[99,149]
    
    yuv_image = cv2.merge([y_img, u_img, v_img])
    return yuv_image

def psnr(img1, img2):

    #Calculates the PSNR between two images

    mse = np.mean((img1 - img2)**2)
    return 10 * np.log10(255 / mse)
    


img=cv2.imread('naturephoto.jpg')
img1=BGR_to_YUV1(img)
img2=YUV_to_BGR1(img1)

img11=BGR_to_YUV(img)
img22=YUV_to_BGR(img1)



y,u,v = downsample(img1)
new_img = upsample(y,u,v)
done = YUV_to_BGR(new_img)

print(img1)
print("break")
print(img11)

imgref= cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
print("break")
print(imgref)
'''
print("break")
print(new_img)
print("break")
print(psnr(done,img))
'''
cv2.imshow('original',img)
cv2.imshow('original converted',imgref)
cv2.imshow('modified',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
