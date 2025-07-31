from PIL import Image
import numpy as np

img = Image.open('cameraman.tif').convert('L')
img_np = np.asarray(img).astype(np.float32)

# Artificially narrow dynamic range for testing
dull_img = (img_np - np.min(img_np)) * (50.0 / (np.max(img_np) - np.min(img_np))) + 100
dull_img = np.clip(dull_img, 0, 255).astype(np.uint8)

Image.fromarray(dull_img).save('dull_cameraman.jpg')

# Now stretch it
min_val = np.min(dull_img)
max_val = np.max(dull_img)
print(f"Simulated Dull Min: {min_val}, Max: {max_val}")

stretched = (dull_img - min_val) * (255.0 / (max_val - min_val))
stretched = np.clip(stretched, 0, 255).astype(np.uint8)

Image.fromarray(stretched).save('20701080_Q14_output.jpg')
