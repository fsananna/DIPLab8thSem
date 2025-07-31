from PIL import Image
import numpy as np

img = Image.open('cameraman.tif').convert('L')
img_np = np.asarray(img)

threshold = 128
binary_img = (img_np > threshold).astype(np.uint8)

label = 1
labeled_img = np.zeros_like(binary_img, dtype=np.int32)
rows, cols = binary_img.shape
stack = []

for i in range(rows):
    for j in range(cols):
        if binary_img[i, j] == 1 and labeled_img[i, j] == 0:
            label += 1
            stack.append((i, j))
            while stack:
                x, y = stack.pop()
                if labeled_img[x, y] == 0 and binary_img[x, y] == 1:
                    labeled_img[x, y] = label
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols:
                                if labeled_img[nx, ny] == 0 and binary_img[nx, ny] == 1:
                                    stack.append((nx, ny))

unique, counts = np.unique(labeled_img, return_counts=True)
remove_labels = unique[counts < 100]

for rl in remove_labels:
    labeled_img[labeled_img == rl] = 0

final_img = (labeled_img > 0).astype(np.uint8) * 255
Image.fromarray(final_img).save('20701080_Q19.jpg')
