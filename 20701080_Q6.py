import cv2
import numpy as np

def custom_sharpen_manual(imgPath, outPutPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image")
        return

    # Define sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    pad = 1
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    sharpened_img = np.zeros_like(img, dtype=np.float32)

    # Manual Convolution Loop
    for i in range(pad, padded_img.shape[0] - pad):
        for j in range(pad, padded_img.shape[1] - pad):
            region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
            value = np.sum(region * kernel)
            sharpened_img[i-pad, j-pad] = value

    # Clip values and convert to uint8
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)

    # Save Output
    cv2.imwrite(outPutPath, sharpened_img)
    print(f"Sharpened image saved as {outPutPath}")

if __name__ == "__main__":
    custom_sharpen_manual('cameraman.tif', '20701080_Q6.jpg')
