from PIL import Image
import numpy as np

img = Image.open('peppers.jpeg')
M = np.asarray(img)

red_channel = np.zeros_like(M)
green_channel = np.zeros_like(M)
blue_channel = np.zeros_like(M)

red_channel[:, :, 0] = M[:, :, 0]
green_channel[:, :, 1] = M[:, :, 1]
blue_channel[:, :, 2] = M[:, :, 2]

combined = np.concatenate((red_channel, green_channel, blue_channel), axis=1)
Image.fromarray(combined).save('20701080_Q12.jpg')
