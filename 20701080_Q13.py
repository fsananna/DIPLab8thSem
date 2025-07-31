from PIL import Image
import numpy as np

img = Image.open('peppers.jpeg')
rgb = np.asarray(img).astype('float32') / 255.0

R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

Cmax = np.max(rgb, axis=2)
Cmin = np.min(rgb, axis=2)
delta = Cmax - Cmin

H = np.zeros_like(Cmax)
mask = delta != 0
idx = (Cmax == R) & mask
H[idx] = (60 * ((G[idx] - B[idx]) / delta[idx])) % 360
idx = (Cmax == G) & mask
H[idx] = (60 * ((B[idx] - R[idx]) / delta[idx])) + 120
idx = (Cmax == B) & mask
H[idx] = (60 * ((R[idx] - G[idx]) / delta[idx])) + 240
H = (H / 360.0) * 255.0

S = np.zeros_like(Cmax)
S[Cmax != 0] = (delta[Cmax != 0] / Cmax[Cmax != 0]) * 255.0

V = Cmax * 255.0

H_img = np.uint8(H)
S_img = np.uint8(S)
V_img = np.uint8(V)

H_color = np.stack([H_img, H_img, H_img], axis=2)
S_color = np.stack([S_img, S_img, S_img], axis=2)
V_color = np.stack([V_img, V_img, V_img], axis=2)

combined = np.concatenate((H_color, S_color, V_color), axis=1)
Image.fromarray(combined).save('20701080_Q13.jpg')
