import cv2
import numpy as np

def salt_pepper_and_median(imgPath, outPutPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image")
        return

    # Added Salt and Pepper Noise
    noisy_img = img.copy()
    prob = 0.05  # Noise density
    rnd = np.random.rand(*img.shape)
    noisy_img[rnd < prob] = 0       # Pepper
    noisy_img[rnd > 1 - prob] = 255 # Salt

    # Applied Median Filter
    denoised_img = cv2.medianBlur(noisy_img, 3)

    # Combine Side-by-Side
    combined = np.hstack((noisy_img, denoised_img))

    # Save Output
    cv2.imwrite(outPutPath, combined)
    print(f"Saved output image as {outPutPath}")

if __name__ == "__main__":
    salt_pepper_and_median('cameraman.tif', '20701080_Q5.jpg')
