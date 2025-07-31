from PIL import Image
import numpy as np

def histogram_equalization(channel):
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    cdf_mapped = np.ma.filled(np.ma.masked_equal(cdf, 0), 0)
    equalized = cdf_normalized[channel]
    return equalized.astype('uint8')

img = Image.open('peppers.jpeg')
rgb = np.asarray(img)

r_eq = histogram_equalization(rgb[:, :, 0])
g_eq = histogram_equalization(rgb[:, :, 1])
b_eq = histogram_equalization(rgb[:, :, 2])

r_img = np.zeros_like(rgb)
g_img = np.zeros_like(rgb)
b_img = np.zeros_like(rgb)

r_img[:, :, 0] = r_eq
g_img[:, :, 1] = g_eq
b_img[:, :, 2] = b_eq

combined = np.concatenate((r_img, g_img, b_img), axis=1)
Image.fromarray(combined).save('20701080_Q15.jpg')
