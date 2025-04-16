import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_beads(image):
    # Convert to float for processing
    image = np.float64(image)

    # Approximate flatfield correction with non-local means denoising
    corrected = cv2.fastNlMeansDenoising(image.astype(np.uint8), None, 30, 7, 21)

    # Apply logarithmic transformation
    log_image = 3 * np.log1p(corrected)

    # Display original and processed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(log_image, cmap='gray')
    plt.title('Log Transformed')
    plt.show()

    # Smooth and threshold image
    blurred = cv2.blur(log_image, (3, 3))
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to clean noise
    kernel = np.ones((5, 5), np.uint8)
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Hough Circle Detection
    circles = cv2.HoughCircles(binary_cleaned, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                param1=50, param2=30, minRadius=30, maxRadius=100)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        centers = circles[:, :2]
        radii = circles[:, 2]

        plt.imshow(log_image, cmap='gray')
        for x, y, r in zip(centers[:, 0], centers[:, 1], radii):
            plt.gca().add_patch(plt.Circle((x, y), r, color='r', fill=False, linewidth=2))
        plt.title('Detected Beads')
        plt.show()

        return centers, radii
    else:
        return [], []
