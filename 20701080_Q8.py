import cv2
import numpy as np

def ideal_low_pass_filter(imgPath, outPutPath, D0=50):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: The image could not be loaded")
        return
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = np.zeros((rows, cols, 2), np.uint8)
    for u in range(rows):
        for v in range(cols):
            if np.sqrt((u - crow)**2 + (v - ccol)**2) <= D0:
                mask[u, v] = 1
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    cv2.imwrite(outPutPath, img_back)
    print(f"Ideal Low Pass Filtered image saved as '{outPutPath}'")

if __name__ == "__main__":
    imgPath = 'cameraman.tif'
    outPutPath = '20701080_Q8.jpg'
    ideal_low_pass_filter(imgPath, outPutPath)
