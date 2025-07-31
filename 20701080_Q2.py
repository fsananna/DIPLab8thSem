import cv2
import numpy as np

def gaussian_filter_manual(imgPath, outPutPath, kernel_size=5, sigma=1.0):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: The image could not be loaded")
        return
    pad = kernel_size // 2
    gaussian_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for x in range(-pad, pad+1):
        for y in range(-pad, pad+1):
            gaussian_kernel[x+pad, y+pad] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    gaussian_kernel /= np.sum(gaussian_kernel)
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    smoothed_img = np.zeros_like(img)
    for i in range(pad, padded_img.shape[0] - pad):
        for j in range(pad, padded_img.shape[1] - pad):
            region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
            smoothed_img[i-pad, j-pad] = np.sum(region * gaussian_kernel)
    cv2.imwrite(outPutPath, smoothed_img)
    print(f"Gaussian filtered image saved as '{outPutPath}'")

if __name__ == "__main__":
    imgPath = 'cameraman.tif'
    outPutPath = '20701080_Q2.jpg'
    gaussian_filter_manual(imgPath, outPutPath)
