import cv2
import numpy as np

def average_filter_manual(imgPath, outPutPath, kernel_size=3):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: The image could not be loaded")
        return
    pad = kernel_size // 2
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    smoothed_img = np.zeros_like(img)
    for i in range(pad, padded_img.shape[0] - pad):
        for j in range(pad, padded_img.shape[1] - pad):
            region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
            smoothed_img[i-pad, j-pad] = np.sum(region) // (kernel_size * kernel_size)
    cv2.imwrite(outPutPath, smoothed_img)
    print(f"Averaging filter applied image saved as '{outPutPath}'")

if __name__ == "__main__":
    imgPath = 'cameraman.tif'
    outPutPath = '20701080_Q1.jpg'
    average_filter_manual(imgPath, outPutPath)
