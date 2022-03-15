""" Visualize the images, compute fourier and wavelets transform, histograms or SIFT"""

import src.util.utils as ut
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import numpy as np
from skimage.feature import hog
import cv2

Xtr,Ytr,Xte = ut.read_data('../../data/')
#im = Xtr[0]
im = Xtr[np.random.randint(0,len(Xtr))] # Select random image
imR, imG, imB = im[:1024].reshape(32,32), im[1024:2048].reshape(32,32), im[2048:].reshape(32,32)

# Plot each of the three components of the image
# fig,ax = plt.subplots(1,3)
# ax[0].imshow(imR)
# ax[1].imshow(imG)
# ax[2].imshow(imB)

# Visualize histograms
# hr, hg, hb = ut.image_to_hist(im,100,-0.1,0.1)
# fig,ax = plt.subplots(1,3)
# ax[0].plot(hr[1][1:],hr[0])
# ax[1].plot(hg[1][1:],hg[0])
# ax[2].plot(hb[1][1:],hb[0])


# Visualize Fourier transform
# fig,ax = plt.subplots(1,3)
# ax[0].imshow(np.abs(fftshift(fft2(imR))))
# ax[1].imshow(np.abs(fftshift(fft2(imG))))
# ax[2].imshow(np.abs(fftshift(fft2(imB))))

# Visualize HOG-features

# fd, hog_image = hog(imR, orientations=8, pixels_per_cell=(8, 8),
#                     cells_per_block=(2, 2), visualize=True)
# plt.figure()
# plt.plot(fd)

# Visualize SIFT
im = np.mean([imR,imG,imB], axis=0)
img = (im - np.min(im)) * 255 / np.max(im)
img = img.astype(np.uint8)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
# draw the detected key points
sift_image = cv2.drawKeypoints(img, keypoints, img)
# show the image
cv2.imshow('image', sift_image)
# save the image
cv2.imwrite("../../brouillons_agathe/imR-sift.jpg", sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Wavelets transform
import numpy as np
import matplotlib.pyplot as plt

import pywt




# Wavelet transform of image, and plot approximation and details
# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']
# coeffs2 = pywt.dwt2(imR, 'bior1.3')
# LL, (LH, HL, HH) = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# fig.tight_layout()
# plt.show()