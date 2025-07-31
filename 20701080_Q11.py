import numpy as np
import cv2

image = cv2.imread('cameraman.tif', 0)
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

mask = np.ones((rows, cols, 2), np.uint8)
notch_radius = 10
notch_center = [(crow+50, ccol+50), (crow-50, ccol-50)]
for center in notch_center:
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if D < notch_radius:
                mask[i, j] = 0

fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_notch = cv2.idft(f_ishift)
img_notch = cv2.magnitude(img_notch[:, :, 0], img_notch[:, :, 1])
cv2.imwrite('20701080_Q11.jpg', cv2.normalize(img_notch, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
