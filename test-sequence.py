import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, filters
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.draw import circle_perimeter
from skimage.morphology import skeletonize, closing, opening, square
from tkinter import filedialog, Tk
import pandas as pd

# Settings
scale = 0.8720
sensitivity_bin = 0.51

# Choose image from directory
root = Tk()
root.withdraw()
myDir = filedialog.askdirectory()  # Select directory
myFiles = [f for f in os.listdir(myDir) if f.endswith('.tif')]

# Read files and create 3D image stack
array = []
for file in myFiles:
    fullFileName = os.path.join(myDir, file)
    img = cv2.imread(fullFileName, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    array.append(img)
array = np.stack(array, axis=-1)

# Select slice and select bead to analyze
slice_selected = 1
plt.imshow(array[:, :, slice_selected - 1], cmap='gray')
plt.title(f'Slice {slice_selected - 1}')
plt.show()

# Crop image interactively
roi = cv2.selectROI("Select Region", array[:, :, slice_selected - 1], fromCenter=False)
loc = (roi[0], roi[1], roi[2], roi[3])

cropped_array = array[loc[1]:loc[1] + loc[3], loc[0]:loc[0] + loc[2], slice_selected - 1]
plt.imshow(cropped_array, cmap='gray')
plt.show()

# Detect the bead and measure its diameter
# Assuming a function `detect_beads` exists for bead detection
def detect_beads(image):
    # Placeholder bead detection function
    # In reality, you should implement the logic to detect beads
    # For simplicity, just a mock function
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=20)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        centers = circles[0, :, :2]
        radii = circles[0, :, 2]
        return centers, radii
    return [], []

centers, radii = detect_beads(cropped_array)
if len(centers) > 0:
    diameter = 2 * (radii / scale)
    pd.DataFrame(diameter).to_csv('diameters.csv', mode='a', header=False)
else:
    print("No beads detected")

# Transform Image to Fourier Space
input_image = cropped_array.astype(np.float64)
j = gaussian_filter(input_image, sigma=60)
J = fftshift(fft2(j))

# Apply Low Pass Filter
low_pass = np.zeros_like(cropped_array)
low_pass_size = cropped_array.shape[0] // 4
low_pass[cropped_array.shape[0] // 2 - low_pass_size: cropped_array.shape[0] // 2 + low_pass_size,
         cropped_array.shape[1] // 2 - low_pass_size: cropped_array.shape[1] // 2 + low_pass_size] = 1
J_lowpass = J * low_pass
j_lowpass = np.abs(ifft2(fftshift(J_lowpass)))

# Plot the images
plt.subplot(1, 3, 1)
plt.imshow(cropped_array, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(j_lowpass, cmap='gray')
plt.title('Flat Field + Low Pass')

plt.subplot(1, 3, 3)
plt.imshow(np.log(np.abs(J_lowpass)), cmap='gray')
plt.title('Fourier Domain')
plt.show()

# Non-local filter
B = np.abs(j_lowpass)
plt.imshow(B, cmap='gray')

# Crop region of interest
# Select ROI interactively
roi = cv2.selectROI("Select Region", B, fromCenter=False)
roi_position = roi
patch = B[roi_position[1]:roi_position[1] + roi_position[3], roi_position[0]:roi_position[0] + roi_position[2]]
patchSq = patch ** 2
edist = np.sqrt(np.sum(patchSq, axis=2))
patchSigma = np.sqrt(np.var(edist))
DoS = 1.5 * patchSigma

# Apply non-local means filter
filtered_im = filters.rank.mean(B, selem=np.ones((5, 5)))  # Replace with an actual non-local filter if needed
plt.imshow(filtered_im, cmap='gray')
plt.title('Filtered Image')
plt.show()

# Binarize
B_f = 1 - filtered_im
BW = B_f > sensitivity_bin

# Morphological operations
BW = closing(BW, square(7))
BW = opening(BW, square(3))
plt.imshow(BW, cmap='gray')
plt.title('After Morphological Operations')
plt.show()

# Skeletonization
skel = skeletonize(BW)
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(skel, cmap='gray')
plt.title('Skeleton')
plt.show()

# Overlay bead
# Create a mask for the bead
[rowsInImage, columnsInImage] = np.meshgrid(np.arange(1, B.shape[0] + 1), np.arange(1, B.shape[1] + 1))
circlePixels = (rowsInImage - centers[0][1]) ** 2 + (columnsInImage - centers[0][0]) ** 2 <= radii[0] ** 2
inner_circle = (rowsInImage - centers[0][1]) ** 2 + (columnsInImage - centers[0][0]) ** 2 <= (radii[0] - 1) ** 2
skel = np.logical(skel + circlePixels) - inner_circle

# Show overlay
plt.imshow(skel, cmap='hot')
plt.title('Bead Overlay')
plt.show()

# Labeling
labeled_image, num_sprouts = measure.label(skel, connectivity=2, return_num=True)
plt.subplot(1, 2, 1)
plt.imshow(1 - labeled_image, cmap='gray')
plt.title('Skeleton')

plt.subplot(1, 2, 2)
plt.imshow(labeled_image, cmap='autumn')
plt.title('Overlay')
plt.show()

# Measure length of sprouts
sprouts = labeled_image - circlePixels
sprouts = sprouts > 0
sprouts_L, num_sprouts = measure.label(sprouts, connectivity=2)
lengths = np.zeros(num_sprouts)
for i in range(1, num_sprouts + 1):
    lengths[i - 1] = np.sum(sprouts_L == i)
lengths = lengths / scale
average_length = np.mean(lengths)
total_length = np.sum(lengths)

# Export results to CSV
filename = myFiles[slice_selected - 1]
date, experiment = filename.split('_')[:2]
slice_num = slice_selected - 1

data = {
    'date': date,
    'experiment': experiment,
    'image_num': len(myFiles),
    'slice': slice_num,
    'diameter': diameter,
    'num_sprouts': num_sprouts,
    'lengths': lengths,
    'average_length': average_length,
    'total_length': total_length,
    'sensitivity_bin': sensitivity_bin
}

df = pd.DataFrame([data])
df.to_excel('sequenced-processions.xlsx', index=False, mode='a')

# Save images
output_image_file = f'Skeleton_{date}_{experiment}_z{slice_num}.png'
cv2.imwrite(output_image_file, labeled_image.astype(np.uint8))

output_mask_file = f'Overlay_{date}_{experiment}_z{slice_num}.png'
cv2.imwrite(output_mask_file, np.uint8(cv2.addWeighted(B, 0.5, labeled_image, 0.5, 0)))