import cv2
import numpy as np

def sharpen_filter_manual(imgPath, outPutPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: The image could not be loaded")
        return
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    pad = 1
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    sharpened_img = np.zeros_like(img)
    for i in range(pad, padded_img.shape[0] - pad):
        for j in range(pad, padded_img.shape[1] - pad):
            region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
            value = np.sum(region * kernel)
            sharpened_img[i-pad, j-pad] = np.clip(value, 0, 255)
    cv2.imwrite(outPutPath, sharpened_img)
    print(f"Sharpened image saved as '{outPutPath}'")

if __name__ == "__main__":
    imgPath = 'cameraman.tif'
    outPutPath = '20701080_Q3.jpg'
    sharpen_filter_manual(imgPath, outPutPath)
