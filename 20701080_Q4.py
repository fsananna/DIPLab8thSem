import cv2
import numpy as np

def sobel_edge_manual(imgPath, outPutPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: The image could not be loaded")
        return
    sobelx_kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])
    sobely_kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])
    pad = 1
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    edge_img = np.zeros_like(img, dtype=np.float32)
    for i in range(pad, padded_img.shape[0] - pad):
        for j in range(pad, padded_img.shape[1] - pad):
            region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
            gx = np.sum(region * sobelx_kernel)
            gy = np.sum(region * sobely_kernel)
            edge_img[i-pad, j-pad] = np.sqrt(gx**2 + gy**2)
    edge_img = np.clip(edge_img, 0, 255).astype(np.uint8)
    cv2.imwrite(outPutPath, edge_img)
    print(f"Sobel edge-detected image saved as '{outPutPath}'")

if __name__ == "__main__":
    imgPath = 'cameraman.tif'
    outPutPath = '20701080_Q4.jpg'
    sobel_edge_manual(imgPath, outPutPath)
