import numpy as np
import cv2

image = cv2.imread('cameraman.tif', 0)
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

gaussian_mask = np.zeros((rows, cols, 2), np.float32)
D0 = 30
for i in range(rows):
    for j in range(cols):
        D = np.sqrt((i - crow)**2 + (j - ccol)**2)
        H = np.exp(-(D**2) / (2 * (D0**2)))
        gaussian_mask[i, j] = (H, H)

fshift_gaussian = dft_shift * gaussian_mask
f_ishift = np.fft.ifftshift(fshift_gaussian)
img_gaussian = cv2.idft(f_ishift)
img_gaussian = cv2.magnitude(img_gaussian[:, :, 0], img_gaussian[:, :, 1])
cv2.imwrite('20701080_Q10.jpg', cv2.normalize(img_gaussian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
