from PIL import Image
import numpy as np

img = Image.open('text.jpeg').convert('L')
img_np = np.asarray(img)

mean_intensity = np.mean(img_np)
std_dev = np.std(img_np)

hist, _ = np.histogram(img_np, bins=256, range=(0, 256))
hist = hist / hist.sum()

entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])

descriptor = {
    'Mean Intensity': mean_intensity,
    'Standard Deviation': std_dev,
    'Entropy': entropy
}

with open('20701080_Q22.txt', 'w') as f:
    for k, v in descriptor.items():
        line = f"{k}: {v}\n"
        print(line.strip())  # Print to console
        f.write(line)
