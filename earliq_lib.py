import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import scipy.misc

def myPlot(image, roi=None,filename=None):
    plt.imshow(image, cmap='gray')
    #magma_cmap = matplotlib.cm.get_cmap('magma')
    if roi is not None:
        roi_cmap = plt.cm.Reds
        
        roi_cmap.set_under(alpha=0)

        plt.imshow(roi, cmap=roi_cmap, alpha=0.5, vmin=0.5);
        #plt.contour(roi, [0.5], colors='r')


        
        
def maskPlot(image, roi=None, overlay_type='contour', **kwargs):
    plt.imshow(image, cmap='gray', interpolation='nearest', **kwargs)

    if roi is not None:
        if overlay_type == 'contour':
            plt.contour(roi, [0.5], colors='r', interpolation='nearest')
        elif overlay_type == 'area':
            roi_cmap = plt.cm.jet
            roi_cmap.set_under(alpha=0)
            plt.imshow(roi, cmap=roi_cmap, alpha=0.5, vmin=0.5, interpolation='nearest')
    

            
def getSphereIndex(data, show=False):
    profile = data.sum(axis=1).sum(axis=1)
    sphere_index = np.where(profile==profile.max())[0][0]
    if show:
        plt.plot(profile); plt.show()
        #print sphere_index
    return sphere_index

from skimage.transform import radon
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate

def getRotation(view, show=False):
    max_angle = 10.
    theta = np.linspace(-max_angle, max_angle, max_angle*100, endpoint=False)
    rad = radon(view, theta=theta, circle=False)
    rad = gaussian_filter(rad, 10)
    profile = rad.max(axis=0)
    
    z = np.polyfit(theta, profile, 2)
    p = np.poly1d(z)
    angle = -z[1]/(2*z[0])
    
    if show:
        plt.figure(figsize=(4,2))
        plt.plot(theta, profile,'y.', theta, p(theta), 'k--', [angle], p([angle]), 'o'); plt.show()    
        #print angle
    return angle

def correctAxis(data, axis, order=1, show=False):
    flat = data.sum(axis=axis)
    threshold = flat.mean()
    angle = getRotation(flat > threshold, show)

    axes = np.delete(np.arange(data.ndim), axis)
    out = rotate(data, -angle, axes=axes, reshape=False)
    
    if show:
        plt.figure(figsize=(12,6))
        plt.subplot(121); plt.imshow(flat > threshold)
        plt.subplot(122); plt.imshow(out.sum(axis=axis) > threshold)
        plt.show()
    return out


def getCenter(data, sigma, show=False):
    sumimg = gaussian_filter(data, sigma).max(axis=0)    

    x_profile = sumimg.max(axis=0)
    y_profile = sumimg.max(axis=1)

    x_center = np.where(x_profile==x_profile.max())[0][0]
    y_center = np.where(y_profile==y_profile.max())[0][0]

    if show:
        plt.figure(figsize=(12,6))
        plt.subplot(121); plt.imshow(sumimg)
        plt.subplot(122); plt.plot(x_profile); plt.plot(y_profile)
        plt.show()
        print (x_center, y_center)
    return x_center, y_center


