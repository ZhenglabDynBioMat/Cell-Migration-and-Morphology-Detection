import cv2
import numpy as np

# Constants
CENTER_WEIGHT = 1
BLUR_RANGE = (5, 5)
BG_THRESHOLD = 120
AREA_THRESHOLD = 300

# Load the TIFF image using OpenCV
img = cv2.imread('test.tif', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image could not be loaded.")
else:
    _, threshold = cv2.threshold(img, BG_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Show the original image
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    # Initialize drawing flag and starting coordinates
    drawing = False
    ix, iy = -1, -1

    # Mouse callback function to draw on the image
    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing
        # Left mouse button press: Start drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        # Mouse move while pressing: Draw a circle
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if drawing:
                cv2.circle(threshold, (x, y), 2, (0, 0, 0), -5)  # Draw black circles
        # Left mouse button release: Stop drawing
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    # Set up mouse callback for drawing
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    # Main loop to display the image and allow user drawing
    while True:
        cv2.imshow('image', threshold)  # Show the updated threshold image
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press
        if key == ord('q'):  # Press 'q' to quit the drawing loop
            break

    # Find contours in the threshold image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Font for displaying text on the image
    font = cv2.FONT_HERSHEY_COMPLEX
    i = 0  # Initialize the counter for bead identification

    # Iterate through each contour detected
    print(f"Number of contours detected: {len(contours)}")
    for cnt in contours:
        area = cv2.contourArea(cnt)  # Calculate area of the contour

        # Skip contours that are too small
        if area < AREA_THRESHOLD:
            continue

        area = CENTER_WEIGHT * area  # Adjust area based on CENTER_WEIGHT

        # Compute the contour's centroid using moments
        M = cv2.moments(cnt)
        if M["m00"] != 0:  # To avoid division by zero
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

        # Compute the perimeter (arc length) of the contour
        p = cv2.arcLength(cnt, False)

        # Calculate the circularity (K) of the contour
        k =  (4 * np.pi * area) / (p * p)

        # Get the minimum enclosing rectangle (min area rectangle)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect  # Get the center, width, height, and angle of the rectangle

        # Calculate the aspect ratio (width / height)
        aspect_ratio = w / h if w > h else h / w  # Ensure the ratio is positive and defined

        # Update bead counter
        i += 1

        # Print the results (bead number, perimeter, area, circularity, aspect ratio)
        print(f"{i}: Perimeter = {int(p)}, Area = {int(area)}, Circularity = {int(k)}, Aspect Ratio = {aspect_ratio:.2f}")

        # Display the bead number, perimeter, area, circularity, and aspect ratio on the image
        text = f"ID: {i}, P: {int(p)}, A: {int(area)}, K: {int(k)}, AR: {aspect_ratio:.2f}"
        cv2.putText(threshold, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (128, 0, 0), 1)

    # Display the thresholded image with annotated bead numbers and other details
    cv2.imshow("Thresholded Image with Beads", threshold)
    cv2.waitKey(0)

    # Clean up and close all windows
    cv2.destroyAllWindows()