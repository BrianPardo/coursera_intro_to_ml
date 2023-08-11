import numpy as np
import matplotlib.pyplot as plt
from numpy import matmul as mm

def closest_centroid(x, centroids):
    """ Return indices of closest centroids for data x """
    idx = np.zeros(len(x)).astype('int64')
    for i, pixel in enumerate(x):
        distance = np.inf
        for j, centroid in enumerate(centroids):
            curr_distance = np.linalg.norm(centroid - pixel)
            if (curr_distance < distance):
                distance = curr_distance
                idx[i] = j
    return idx

def update_centroids(x, index, k):
    """ find new centroids based on average of assigned points """
    centroids = np.zeros((k, np.shape(x)[1]))
    
    for i in range(k):
        p = (index == i).reshape(-1,1)
        centroids[i,:] = mm(p.T, x)/np.sum(p)
    
    return centroids

def run_kmeans(x, k, init_centroids, iters):
    """ Run K-means algorithm on a dataset for k clusters """
    centroids = init_centroids
    
    for i in range(iters):
        if iters!=1:
            print(f'Iteration {i+1}')
        # find closest centroid
        idx = closest_centroid(x, centroids)
        
        # update centroid
        centroids = update_centroids(x, idx, k)
    
    return idx, centroids

def initialize_centroids(x, k):
    """ Choose k random points as centroids from data x """
    # number of examples
    m = len(x)
    
    # initialize random centroids
    idx = np.random.choice(range(m), k, replace = False)
    
    return x[idx]

def display_data(x):
    """ Prepare set of images to display in a grid """
    # m is number of samples, n is number of pixels
    (m, n) = np.shape(x)
    
    # number of rows/columns of pixels and images
    pad = 1
    l_pix = int(np.floor(np.sqrt(n)))
    l_img = int(np.ceil(np.sqrt(m)))
    w = l_pix*l_img+l_img*pad-1
    
    # initialize the grid
    img_grid = np.ones((w,w))
    
    for i, img in enumerate(x):
        # compute the start and end pixel for each image w/ pad
        h_start = (i%l_img)*l_pix+pad*(i%l_img)
        h_end = h_start+l_pix
        v_start = ((i)//l_img)*l_pix+pad*(i//l_img)
        v_end = v_start+l_pix
        
        # reshape image into square and replace image in grid
        image = img.reshape(l_pix, l_pix).T
        image = img
        img_grid[v_start:v_end, h_start:h_end] = image
        
    return img_grid

def display_data(x, figsize):
    """ Prepare set of images to display in a grid """
    # m is number of samples, n is number of pixels
    (m, n) = np.shape(x)
    
    # number of rows/columns of pixels and images
    pad = 0.1
    ex_width = int(np.floor(np.sqrt(n)))
    fig_cols = int(np.ceil(np.sqrt(m)))
    fig_rows = int(np.ceil(m/fig_cols))
    
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize)
    fig.subplots_adjust(wspace=pad, hspace=pad)

    axes = [axes] if m == 1 else axes.ravel()
    for i in range(len(axes)):
        axes[i].imshow(x[i].reshape(ex_width, ex_width, order='F'),cmap='gray'); axes[i].axis('off')


# This displays data stored in X in a 2D grid:
def displaydata(X, figsize):
    # Determines the number of rows, cols of the entire figure:
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        m = 1; n = X.size
        X = X[None]  # transforms X into a 1xn array
    else:
        raise IndexError('The input X should be a 1 or 2 dimensional numpy array.')
    # For each individual example to be displayed, we have to determine its dimensions:
    ex_width = int(np.round(np.sqrt(n))); ex_height = int(np.round(n/ex_width))        
    # Determines the number of items to be displayed in the figure:
    fig_rows = int(np.floor(np.sqrt(m))); fig_cols = int(np.ceil(m /fig_rows))
    # Creates a figure (fig) with an array (ax_array) of subplots:
    fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize)
    # Adjust spacing between subplots:
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()
    for i in range(len(ax_array)):
        ax_array[i].imshow(X[i].reshape(ex_width, ex_height, order='F'),cmap='gray'); ax_array[i].axis('off')


def normalize(x):
    """ Normalize data to 0 mean and unit standard deviation """
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=1)
    return (x-mu)/std, mu, std

def pca(x):
    m = len(x)
    Sigma = (1/m)*mm(x.T, x)
    U, S, _ = np.linalg.svd(Sigma)
    return U, S

def project_data(x, U, k): 
    return mm(x, U[:,:k])

def recover_data(z, U, k):
    return mm(z, U[:,:k].T)