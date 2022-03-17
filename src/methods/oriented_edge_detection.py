""" Implement oriented edge energy filtering """

from scipy.ndimage.filters import gaussian_filter
import numpy as np
import src.util.utils as ut
import matplotlib.pyplot as plt
from scipy import signal

def f1(x, y, sigma, l, C):
    trmy = 2/sigma**2 * (2/sigma**2*y**2-1)*np.exp(-y**2/sigma**2)
    return trmy/C*np.exp(-x**2/(l**2*sigma**2))

def rotate_f1(x,y,sigma,l,C, theta):
    x_ = np.cos(theta)*x + np.sin(theta)*y
    y_ = -np.sin(theta)*x + np.cos(theta)*y
    return f1(x_,y_,sigma,l,C)

def create_filters(nb,sigma,l,C,size,born):
    grid = np.linspace(-born, born, size)
    x,y = np.meshgrid(grid, grid)

    filters = []
    step = np.pi/nb
    angle = 0
    for i in range(nb):
        z = rotate_f1(x,y,sigma,l,C,angle)
        filters.append(z.copy())
        angle+=step
    return filters



def apply_filters(img, filters):

    transformed_images= []
    for z in filters:
        img_ = signal.convolve2d(img, z, boundary='symm', mode='same')
        transformed_images.append(img_)
    return transformed_images


def plot_histograms(transformed_images, bins = 12, bound = -0.5):
    fig, ax = plt.subplots(1,len(transformed_images))
    for i,z in enumerate(transformed_images):
        ax[i].hist(z.flatten(),bins = bins)
    plt.savefig('histograms.png')
    return

def energy_histograms(transformed_images, bins,bound = 0.5):
    histograms = []
    for z in transformed_images:
        h = np.histogram(z.flatten(),bins,range=[-bound, bound])[0]
        histograms.append(h)
    return np.concatenate(histograms)

def image_to_energy_hist(img,filters, bins, bound):
    transformed_images = apply_filters(img, filters)
    return energy_histograms(transformed_images, bins, bound)

def Xtr_to_energy_hist(Xtr, filters, bins, bound):
    features = []
    for im in Xtr:
        imR, imG, imB = im[:1024].reshape(32, 32), im[1024:2048].reshape(32, 32), im[2048:].reshape(32, 32)
        featuresR = image_to_energy_hist(imR, filters, bins, bound)
        featuresG = image_to_energy_hist(imG, filters, bins, bound)
        featuresB = image_to_energy_hist(imB, filters, bins, bound)
        features.append(np.concatenate([featuresR, featuresG, featuresB]))
    return np.array(features)







def main():
    Xtr,Ytr,Xte = ut.read_data('../../data/')
    im = Xte[2]
    # im = Xtr[np.random.randint(0,len(Xtr))] # Select random image
    imR, imG, imB = im[:1024].reshape(32,32), im[1024:2048].reshape(32,32), im[2048:].reshape(32,32)
    gray = np.mean([imR, imG, imB], axis=0)

    plt.figure()
    plt.imshow(gray, cmap='gray')
    plt.savefig('gray_img.png')

    filters = create_filters(8,1,0.5,1,5,1)
    fig, ax = plt.subplots(1, len(filters))
    for i,z in enumerate(filters):
        ax[i].imshow(z)
    plt.savefig('filters.png')

    # Apply the filters
    energy_images = apply_filters(gray,filters)

    fig, ax = plt.subplots(1, len(filters))
    for i,z in enumerate(energy_images):
        ax[i].imshow(z)
    plt.savefig('energy.png')

    plot_histograms(energy_images)

    histograms = energy_histograms(energy_images,12)

    features = Xtr_to_energy_hist(Xtr, filters, 15, 0.5)
    print(features.shape)


if __name__ == "__main__":
    main()
