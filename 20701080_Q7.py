import cv2
import numpy as np

image = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

fourier = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

fourier_shift = np.fft.fftshift(fourier)

magnitude = 20 * np.log(cv2.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))

magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

cv2.imwrite('20701080_Q7.jpg', magnitude)
