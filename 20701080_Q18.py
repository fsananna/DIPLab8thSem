from PIL import Image
import numpy as np

img = Image.open('cameraman.tif').convert('L')
img_np = np.asarray(img)

threshold = 128
binary_img = (img_np > threshold).astype(np.uint8) * 255

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.uint8)

pad = 1
padded_img = np.pad(binary_img, pad, mode='constant', constant_values=0)
dilated = np.zeros_like(binary_img)

for i in range(pad, padded_img.shape[0] - pad):
    for j in range(pad, padded_img.shape[1] - pad):
        region = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1]
        if np.any(region == 255):
            dilated[i - pad, j - pad] = 255

padded_dilated = np.pad(dilated, pad, mode='constant', constant_values=0)
closed_img = np.zeros_like(binary_img)

for i in range(pad, padded_dilated.shape[0] - pad):
    for j in range(pad, padded_dilated.shape[1] - pad):
        region = padded_dilated[i - pad:i + pad + 1, j - pad:j + pad + 1]
        if np.all(region == kernel * 255):
            closed_img[i - pad, j - pad] = 255

Image.fromarray(closed_img).save('20701080_Q18.jpg')
