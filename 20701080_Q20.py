from PIL import Image
import numpy as np

img = Image.open('coins.jpeg')
img_np = np.asarray(img).reshape((-1, 3))
K = 3

centroids = img_np[np.random.choice(img_np.shape[0], K, replace=False)]
for _ in range(10):
    distances = np.linalg.norm(img_np[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    for k in range(K):
        if np.any(labels == k):
            centroids[k] = img_np[labels == k].mean(axis=0)
        else:
            centroids[k] = img_np[np.random.choice(img_np.shape[0])]

segmented_img = centroids[labels].reshape((img.size[1], img.size[0], 3)).astype(np.uint8)
Image.fromarray(segmented_img).save('20701080_Q20.jpg')
