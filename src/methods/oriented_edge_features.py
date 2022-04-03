""" Implement oriented edge energy filtering """

from scipy.ndimage.filters import gaussian_filter
import numpy as np
import src.util.utils as ut
import matplotlib.pyplot as plt
from scipy import signal, misc


###############################################################
######## FUNCTIONS TO CREATE FILTERS TO EXTRACT ENERGY FEATURES
###############################################################

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


###################################################################
##### A FIRST CLASS TO REPRESENT IMAGES WITH ENERGY HISTOGRAMS ####

class energy_hist():

    def __init__(self, filters, nbins, bound):
        self.filters = filters
        self.nbins = nbins
        self.bound = bound # the bound for the histograms : [-bound, bound] is separated into nbins segments

    def transform(self, img):
        transformed_images = apply_filters(img, self.filters) # list of energy filtered images
        histograms = []
        for z in transformed_images:
            h = np.histogram(z.flatten(), self.nbins, range=[-self.bound, self.bound], density=True)[0]
            histograms.append(h)
        return np.concatenate(histograms)


    def transform_rgb(self,im):
        imR, imG, imB = im[:1024].reshape(32, 32), im[1024:2048].reshape(32, 32), im[2048:].reshape(32, 32)
        featuresR = self.transform(imR)
        featuresG = self.transform(imG)
        featuresB = self.transform(imB)
        return np.concatenate([featuresR, featuresG, featuresB])

    def transform_all(self,Xtr):
        features = []
        for im in Xtr:
            features.append(self.transform_rgb(im))
        return np.array(features)




###########################################################################
#### ANOTHER CLASS IMPLEMENTING MULTILEVEL ORIENTED ENERGY FEATURES########
# based on http://acberg.com/papers/mbm08cvpr.pdf

def norm16(img):
    """Compute the sum of energy response in each 16x16 subsquare"""
    img_grid = img.reshape(img.shape[0]//16,16,img.shape[1]//16,16)
    img_grid = img_grid.transpose(0, 2, 1, 3)
    norm_factors = np.sum(np.abs(img_grid),axis=(2,3))
    return norm_factors

def normalize16(energy_images):
    '''Normalizes accross directions each subsquare of size 16x16 in the image
    energy_images is of shape (nb_filter*32,32)'''

    norms_factor = norm16(energy_images) # shape nb_filter*2,2
    norms_factor = np.abs(norms_factor.reshape((-1, 2, 2)))
    # Sum accross filters
    norms_factor = np.sum(norms_factor, axis=0) # shape 2,2
    norms_factor = np.kron(norms_factor,np.ones((16,16))) # shape 32,32
    # normalize each image with this factor
    return energy_images.reshape(-1,32,32)/norms_factor

def tile(img, sz):
    """

    :param img: the image to tile
    :param sz: the size of the tiles
    :return: all the tiles in the image. shape (-1, sz,sz)
    """
    img_grid = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz).copy()
    tiles = img_grid.transpose(0, 2, 1, 3).reshape(-1,sz,sz)
    return tiles

def non_max_suppression(img,angle):

    dict_conv = [(np.array([[0,0,0],[0,1,0],[0,-1,0]]),np.array([[0,-1,0],[0,1,0],[0,0,0]])),
                 (np.array([[0, 0, 0], [0, 1, 0], [-1/2, -1/2, 0]]), np.array([[0, -1/2, -1/2], [0, 1, 0], [0, 0, 0]])),
                 (np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]), np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])),
                 (np.array([[0, 0, 0], [-1/2, 1, 0], [-1/2, 0, 0]]), np.array([[0, 0, -1/2], [0, 1, -1/2], [0, 0, 0]])),
                 (np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])),
                 (np.array([[-1/2, 0, 0], [-1/2, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, -1/2], [0, 0, -1/2]])),
                 (np.array([[-1 , 0, 0], [0, 1, 0], [0, 0, 0]]),np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])),
                 (np.array([[-1/2, -1/2, 0], [0, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, 0], [0, -1/2, -1/2]])),
                 ]
    conv1, conv2 = dict_conv[angle]
    img_bool = signal.convolve2d(img, conv1, boundary='symm', mode='same') > 0
    img_bool *= signal.convolve2d(img, conv2, boundary='symm', mode='same') >0

    # print(np.sum(img_bool)/(img.shape[0]*img.shape[1]))
    return img*img_bool



class multi_level_energy_features():

    def __init__(self, size_min,filters, nbins, bound, gray = False, sum = True):

        self.size_min = size_min
        self.gray = gray # Compute the energy response on the gray level image or on each color individually
        self.nb_levels = int(np.log2(32/size_min)) + 1
        self.filters = filters
        self.tile_sizes = [32//2**i for i in range(self.nb_levels)]
        self.weights = [1/(2**self.nb_levels)] + [1/(2**self.nb_levels - i + 1) for i in range(1,self.nb_levels)]
        self.sum = sum
        self.nbins = nbins
        self.bound = bound

    def transform(self,image):
        ''' Return the multi-level energy histogram representation of the array (32,32) image'''

        # First apply the energy filters
        energy_img = apply_filters(image,self.filters) # Big image of size (32*nb_filters ,32)

        # Apply non max suppression
        for i,im in enumerate(energy_img):
            energy_img[i] = non_max_suppression(im**2,i)
            # energy_img[i] = im**2

        energy_img = np.concatenate(energy_img)

        # Then normalize in each 16*16 squares accross directions
        energy_img = normalize16(energy_img).reshape(-1,32)



        # Then tile the image and sum the normalized response in each tile
        level_hist = []
        for i,sz in enumerate(self.tile_sizes):
            tiles = tile(energy_img,sz)
            # histograms = 1/(4**(self.nb_levels - i))* np.sum(tiles, axis=(1,2))
            if self.sum :
                histograms = self.weights[i]*np.sum(np.abs(tiles), axis=(1,2 ))
                level_hist.append(histograms)
            else :
                for t in tiles:
                    level_hist.append(np.histogram(np.abs(t), self.nbins, range=[-self.bound, self.bound], density=True)[0])

        return np.concatenate(level_hist)

    def transform_rgb(self,im):
        """Compute the level_histograms representation for each channel and concatenate"""
        imR, imG, imB = im[:1024].reshape(32, 32), im[1024:2048].reshape(32, 32), im[2048:].reshape(32, 32)

        if self.gray:
            features = self.transform(imR + imG + imB)
            return features

        else:
            featuresR = self.transform(imR)
            featuresG = self.transform(imG)
            featuresB = self.transform(imB)
            return np.concatenate([featuresR,featuresG,featuresB])

    def transform_all(self, Xtr):
        """ Transform all the images in Xtr"""
        all_features = []
        for im in Xtr:
            all_features.append(self.transform_rgb(im))
        return np.array(all_features)


##########################################################################
##### BAG OF HISTOGRAMS FEATURES

class bag_of_hist():

    def __init__(self, patch_size, stride, bins, bound, filters):
        self.patch_size = patch_size
        self.stride = stride
        self.bins = bins
        self.bound = bound
        self.filters = filters

    def return_all_tiles(self, img):
        tiles = []
        i = 0
        while i + self.patch_size <= img.shape[0]:
            j = 0
            while j + self.patch_size <= img.shape[1]:
                tiles.append(img[i:i+self.patch_size,j:j+self.patch_size].copy())
                j += self.stride
            i+=self.stride

        return tiles

    def transform(self,img):
        """returns a bag of patch features for the image"""

        transformed_images = apply_filters(img,self.filters)

        #Extract all tiles
        transformed_tiles = []
        for en_img in transformed_images:
            transformed_tiles.append(self.return_all_tiles(en_img))

        #For each tile associate the feature vector
        bag_features = []
        for i in range(len(transformed_tiles[0])):
            features = []
            for j in range(len(transformed_tiles)):
                features.append(np.histogram(transformed_tiles[j][i], self.bins, [-self.bound, self.bound], density = True)[0])
            features = np.concatenate(features)
            bag_features.append(features)

        return np.array(bag_features)

    def transform_rgb(self,im):
        imR, imG, imB = im[:1024].reshape(32, 32), im[1024:2048].reshape(32, 32), im[2048:].reshape(32, 32)

        featuresR = self.transform(imR)
        featuresG = self.transform(imG)
        featuresB = self.transform(imB)

        features = np.concatenate((featuresR, featuresG,featuresB), axis=1)
        print(features.shape)
        return features


###########################################################################
########### TEST AND VISUALIZE ############################################



def main():
    Xtr,Ytr,Xte = ut.read_data('../../data/')
    im = Xte[1]
    # im = Xtr[np.random.randint(0,len(Xtr))] # Select random image
    imR, imG, imB = im[:1024].reshape(32,32), im[1024:2048].reshape(32,32), im[2048:].reshape(32,32)
    gray = np.mean([imR, imG, imB], axis=0)

    # gray = misc.ascent()
    # Visualize gray image
    plt.figure()
    plt.imshow(gray, cmap='gray')
    plt.savefig('gray_img.png')

    # Create and visualize filters
    filters = create_filters(8,1,3,1,3,1)
    print(filters[0])
    fig, ax = plt.subplots(1, len(filters))
    for i,z in enumerate(filters):
        ax[i].imshow(z)
    plt.savefig('filters.png')

    # Visualize the effect on the image
    transformed_images = apply_filters(gray,filters)
    fig, ax = plt.subplots(1, len(filters))
    for i,z in enumerate(transformed_images):
        ax[i].imshow(z**2)
    plt.savefig('energy.png')

    # Apply non max suppression and visualize
    fig, ax = plt.subplots(1, len(filters))
    for i,img in enumerate(transformed_images):
        z = non_max_suppression(img**2,i)
        ax[i].imshow(z)
    plt.savefig('nms.png')

    # Create a transformation instance MLEF
    mlef = multi_level_energy_features(16,filters, 15, 0.5)
    features = mlef.transform_rgb(im)
    print(features.shape)
    plt.figure(); plt.plot(features); plt.savefig('mlef.png')

    # Create an energy histogram instance
    # en_hist = energy_hist(filters,nbins=30,bound=0.5)
    # features2 = en_hist.transform_rgb(im)
    # plt.figure(); plt.plot(features2); plt.savefig('en_hist.png')

    # Extract bag of features
    # bag_transform = bag_of_hist(16,16,5,0.5,filters)
    # bag_of_features = bag_transform.transform_rgb(im)
    # # Visualize
    # fig, ax = plt.subplots(len(bag_of_features),1)
    # for i,patch in enumerate(bag_of_features):
    #     ax[i].plot(bag_of_features[i])
    # plt.savefig('../../results/bagof.png')



if __name__ == "__main__":
    main()