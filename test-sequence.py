import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, filters
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize, closing, opening, square
from tkinter import filedialog, Tk

# Configuration
scale = 0.5
sensitivity_threshold = 0.51

# Select directory with TIFF images
root = Tk()
root.withdraw()
directory = filedialog.askdirectory()
file_list = [f for f in os.listdir(directory) if f.endswith('.tif')]

# Load image stack into a 3D array
image_stack = [cv2.imread(os.path.join(directory, f), cv2.IMREAD_GRAYSCALE) for f in file_list]
image_stack = np.stack(image_stack, axis=-1)

# Select and preview image slice
slice_index = 1
plt.imshow(image_stack[:, :, slice_index - 1], cmap='gray')
plt.title(f'Slice {slice_index - 1}')
plt.show()

# Crop region of interest
roi = cv2.selectROI("Select Region", image_stack[:, :, slice_index - 1], fromCenter=False)
x, y, w, h = roi
cropped = image_stack[y:y + h, x:x + w, slice_index - 1]
plt.imshow(cropped, cmap='gray')
plt.show()

# Basic bead detection using Hough Transform
def detect_beads(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=5, maxRadius=20)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :, :2], circles[0, :, 2]
    return [], []

centers, radii = detect_beads(cropped)
if centers:
    diameters = 2 * (radii / scale)
    pd.DataFrame(diameters).to_csv('diameters.csv', mode='a', header=False)
else:
    print("No beads detected.")

# Fourier Transform and Low Pass Filtering
image = cropped.astype(np.float64)
blurred = gaussian_filter(image, sigma=60)
fft_image = fftshift(fft2(blurred))

low_pass_filter = np.zeros_like(cropped)
filter_size = cropped.shape[0] // 4
center = (cropped.shape[0] // 2, cropped.shape[1] // 2)
low_pass_filter[center[0]-filter_size:center[0]+filter_size,
                center[1]-filter_size:center[1]+filter_size] = 1

filtered_fft = fft_image * low_pass_filter
filtered_image = np.abs(ifft2(fftshift(filtered_fft)))

# Display image processing steps
plt.subplot(1, 3, 1)
plt.imshow(cropped, cmap='gray')
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Low Pass')

plt.subplot(1, 3, 3)
plt.imshow(np.log(np.abs(filtered_fft)), cmap='gray')
plt.title('Fourier')
plt.show()

# Non-local filtering (placeholder)
B = np.abs(filtered_image)
roi = cv2.selectROI("Select Region", B, fromCenter=False)
patch = B[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
patch_sigma = np.sqrt(np.var(patch ** 2))
DoS = 1.5 * patch_sigma

filtered = filters.rank.mean(B.astype(np.uint8), selem=np.ones((5, 5)))
plt.imshow(filtered, cmap='gray')
plt.title('Filtered')
plt.show()

# Binarization and Morphological Processing
binary = 1 - filtered
BW = binary > sensitivity_threshold
BW = closing(BW, square(7))
BW = opening(BW, square(3))

plt.imshow(BW, cmap='gray')
plt.title('Binary Morphology')
plt.show()

# Skeletonization
skeleton = skeletonize(BW)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(skeleton, cmap='gray')
plt.title('Skeleton')
plt.show()

# Overlay bead detection on skeleton
rows, cols = np.meshgrid(np.arange(B.shape[0]), np.arange(B.shape[1]), indexing='ij')
circle_mask = (rows - centers[0][1]) ** 2 + (cols - centers[0][0]) ** 2 <= radii[0] ** 2
inner_mask = (rows - centers[0][1]) ** 2 + (cols - centers[0][0]) ** 2 <= (radii[0] - 1) ** 2
overlay = np.logical(skeleton + circle_mask) - inner_mask

plt.imshow(overlay, cmap='hot')
plt.title('Bead Overlay')
plt.show()

# Labeling and Measurement
labeled, num_sprouts = measure.label(overlay, connectivity=2, return_num=True)

plt.subplot(1, 2, 1)
plt.imshow(1 - labeled, cmap='gray')
plt.title('Skeleton')

plt.subplot(1, 2, 2)
plt.imshow(labeled, cmap='autumn')
plt.title('Overlay')
plt.show()

# Sprout Length Measurement
sprout_mask = (labeled > 0) & (~circle_mask)
labeled_sprouts, num_sprouts = measure.label(sprout_mask, connectivity=2, return_num=True)
lengths = np.array([np.sum(labeled_sprouts == i) for i in range(1, num_sprouts + 1)]) / scale

# Output Data
filename = file_list[slice_index - 1]
date, experiment = filename.split('_')[:2]
output_data = {
    'date': date,
    'experiment': experiment,
    'image_num': len(file_list),
    'slice': slice_index - 1,
    'diameter': diameters,
    'num_sprouts': num_sprouts,
    'lengths': lengths,
    'average_length': np.mean(lengths),
    'total_length': np.sum(lengths),
    'sensitivity_bin': sensitivity_threshold
}
pd.DataFrame([output_data]).to_excel('sequenced-processions.xlsx', index=False, mode='a')

# Save labeled and overlay images
cv2.imwrite(f'Skeleton_{date}_{experiment}_z{slice_index - 1}.png', labeled.astype(np.uint8))
cv2.imwrite(f'Overlay_{date}_{experiment}_z{slice_index - 1}.png',
            np.uint8(cv2.addWeighted(B, 0.5, labeled, 0.5, 0)))
