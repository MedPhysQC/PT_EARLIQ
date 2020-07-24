import numpy as np
from datetime import datetime

import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt

import skimage
from skimage import feature
from skimage import filters
from skimage.filters import threshold_otsu

import scipy.misc
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_opening
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate, zoom
from scipy.optimize import curve_fit, minimize

from PIL import Image

import pydicom as dcm

plt.rcParams['image.cmap'] = 'jet'


SPHERES_MM = [10, 13, 17, 22, 28, 37]
SPHERES_ML = [4/3.*np.pi*(0.1*d/2.)**3 for d in SPHERES_MM]
LUNG_INSERT_RADIUS = 15

def loadData(dcmfiles):

    
    
    data = []
    position = []
    for fn in dcmfiles:
        ds = dcm.read_file(fn)
        data.append(ds.pixel_array.astype(np.float64) * ds.RescaleSlope)
        position.append(float(ds.ImagePositionPatient[2]))
    position = np.array(position)
    sorting = np.argsort(position)
    position = position[sorting]
    data = np.array(data)[sorting]
    pixsize = [float(x) for x in ds.PixelSpacing] + [(position[-1]-position[0])/(len(position)-1)]

    pixsize = np.abs(np.array(pixsize)[::-1])
    return data, pixsize
        
        
def maskPlot(image, roi=None, overlay_type='contour', **kwargs):
    plt.imshow(image, cmap='gray', interpolation='nearest', **kwargs)

    if roi is not None:
        if overlay_type == 'contour':
            plt.contour(roi, [0.5], colors='r')
        elif overlay_type == 'area':
            roi_cmap = plt.cm.jet
            roi_cmap.set_under(alpha=0)
            plt.imshow(roi, cmap=roi_cmap, alpha=0.5, vmin=0.5, interpolation='nearest')

                
class Activity(object):
    def __init__(self, activity, time, halflife_seconds):
        self.A = activity
        self.t = time
        self.halflife = halflife_seconds
    
    def At(self, t1):
        assert isinstance(t1, datetime)
        return self.A * 0.5**((t1 - self.t).total_seconds()/self.halflife)
        
    def _dA(self, x):
        if isinstance(x, Activity):
            return x.A * 0.5**((self.t - x.t).total_seconds()/self.halflife)
        else:
            return float(x)
    
    def __add__(self, x):
        return Activity(self.A + self._dA(x), self.t)
    
    def __sub__(self, x):
        return Activity(self.A - self._dA(x), self.t)
        



'''
def getCenterCrude(data, sigma, show=False):
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
'''

'''
def getSphereIndex(data, show=False):
    profile = data.sum(axis=1).sum(axis=1)
    sphere_index = np.where(profile==profile.max())[0][0]
    if show:
        plt.plot(profile); plt.show()
        print (sphere_index)
    return sphere_index
'''

def findAngleEdgeDetection(image, show):
    img_edges = feature.canny(image, sigma=0, low_threshold=0.5)
        
    # Classic straight-line Hough transform
    h, theta, d = skimage.transform.hough_line(img_edges)

    if show:
        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(img_edges, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(np.log(1 + h),
                     extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                     cmap='gray', aspect=1/1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(img_edges, cmap='gray')
        for _, angle, dist in zip(*skimage.transform.hough_line_peaks(h, theta, d)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')
        plt.show()
    
    _, angles, _ = skimage.transform.hough_line_peaks(h, theta, d)
    
    angles *= 180./np.pi
    for i,angle in enumerate(angles):
        if angle > 45: angles[i] -= 90.
        if angle < -45: angles[i] += 90.
    
    return np.mean(angles)


def rotatePhantom(data, pixsize, show):
    for axis in (1, 2):
        axes = np.delete(range(data.ndim), axis)
        flat = zoom(data.mean(axis=axis), 4, order=1)
                
        flat = flat > skimage.filters.threshold_otsu(flat)

        angle = findAngleEdgeDetection(flat, show)
        
        data = rotate(data, angle, axes=axes, reshape=False)
        print ("Rotated by %.2f deg. about axis %i"%(angle, axis))
        if show:            
            plt.figure(figsize=(12,6))
            plt.subplot(121); plt.imshow(flat)
            plt.subplot(122); plt.imshow(rotate(flat.astype(float), angle, reshape=False))
            plt.show()
    
    return data

'''
def cropCrude(data, pixsize, show):
    com = center_of_mass(data)
    
    phantom_size = [200., 224., 294.] #Bounding-box dimensons of phantom in mm (z,y,x)
    margin = 100.
    
    if show:
        plt.figure(figsize=(12,6))
        plt.subplot(121); plt.imshow(data.max(axis=0))
        plt.subplot(122); plt.imshow(data.max(axis=1))
        plt.show()
    
    for axis in range(3):
        crop_size = 0.5*(phantom_size[axis]+margin) / pixsize[axis]
            
        i_start = int(com[axis] - crop_size)
        i_end =   int(com[axis] + crop_size)
        
        slc = [slice(None)] * len(data.shape)
        slc[axis] = slice(max(0, i_start), min(data.shape[axis], i_end))
        
        data = data[tuple(slc)]

    if show:
        plt.figure(figsize=(12,6))
        plt.subplot(121); plt.imshow(data.max(axis=0))
        plt.subplot(122); plt.imshow(data.max(axis=1))
        plt.show()
    
    return data
'''

def cropPhantom(data, pixsize, show):
    z_profile = data.mean(axis=1).mean(axis=1)
    sphere_index = np.where(z_profile==z_profile.max())[0][0]
    dist_pix = 40/pixsize[0]
            
    if sphere_index < data.shape[0]/2:
        z_thres = int(sphere_index + dist_pix)
    else:
        z_thres = int(sphere_index - dist_pix)
        
    filtered = gaussian_filter(data, 5.0/pixsize)
        
    otsu_thres = skimage.filters.threshold_otsu(filtered[z_thres])
    bgr_threshold = min(filtered[z_thres].max()/5.0, otsu_thres)
        
    newmask = binary_opening((filtered > bgr_threshold), iterations=5, structure=np.ones((3,3,3)))
    newmask = binary_dilation(newmask, iterations=2)

    in_phantom = np.where(newmask)

    slice_z = slice(in_phantom[0].min(), in_phantom[0].max())
    slice_y = slice(in_phantom[1].min(), in_phantom[1].max())
    slice_x = slice(in_phantom[2].min(), in_phantom[2].max())
    
    crop = data[slice_z, slice_y, slice_x]

    if show:
        plt.figure(figsize=(12,6))
        plt.subplot(131); plt.imshow(crop.max(axis=0))
        plt.subplot(132); plt.imshow(crop.max(axis=1))
        plt.subplot(133); plt.imshow(crop.max(axis=2))
        plt.show()

    return crop


def getSphereAndBackgroundSlices(data, pixsize, show):
    z_profile = data.mean(axis=1).mean(axis=1)

    sphere_index = np.where(z_profile==z_profile.max())[0][0]
    if sphere_index > len(z_profile)/2:
        background_index = int(sphere_index/2.0)
    else:
        background_index = int((len(z_profile) + sphere_index)/2.0)
        
    width_of_sphere_slice = 30. #mm
    width = int(np.ceil( width_of_sphere_slice / pixsize[0] ))

    z_spheres = np.s_[sphere_index-width:sphere_index+width]
    
    z_backgr = np.s_[background_index-width:background_index + width]

    if show:
        plt.plot(z_profile)

        plt.axvline(z_spheres.start, color='r', linestyle='--')
        plt.axvline(sphere_index, color='r')
        plt.axvline(z_spheres.stop, color='r', linestyle='--')

        plt.axvline(z_backgr.start, color='g', linestyle=':')
        plt.axvline(z_backgr.stop, color='g', linestyle=':')

        plt.show()
    
    return z_spheres, z_backgr

'''
def getCenter1D(profile, pixsize, show):
    x_range = np.arange(len(profile))
    valid_indices = np.where(profile)[0]
    fit_profile = profile[valid_indices]
    
    fit_range = x_range[valid_indices]

    p0 = [profile.min()-profile.max(), len(profile)/2, 20/pixsize, profile.max(), 0]
    coeff, var_matrix = curve_fit(gauss, fit_range, fit_profile, p0=p0)
    
    if show:
        plt.plot(x_range, profile)
        #plt.plot(x_range, gauss(x_range, *p0))
        plt.plot(fit_range, gauss(fit_range, *coeff)); plt.show()

    return coeff[1]
'''

def getSphereMask2D(shape, cy, cx, rx, ry):
    dy, dx = shape
    y, x = np.ogrid[-cy:dy-cy, -cx:dx-cx]
    return ((x/rx)**2 + (y/ry)**2 <= 1)

'''
def getCenter(data, pixsize, z_spheres, show):
    arr = np.ma.array(data, mask=False)
    arr.mask[z_spheres] = True
    
    bgr_mean = arr.mean(axis=0)

    cy, cx = np.array(bgr_mean.shape)/2
    ry, rx = 100/pixsize[1:]
    mask = getSphereMask2D(bgr_mean.shape, cy, cx, rx, ry)

    bgr_mean = gaussian_filter(bgr_mean, sigma=10/pixsize[1:])
    bgr_mean = np.ma.array(bgr_mean, mask = ~mask)

    if show: plt.imshow(bgr_mean); plt.show()

    halfwidth = 20/pixsize

    bgr_mean_slice = bgr_mean[int(cy-halfwidth[1]):int(cy+halfwidth[2])]
    if show: plt.imshow(bgr_mean_slice); plt.show()
    profile = bgr_mean_slice.mean(axis=0)
    x_center = getCenter1D(profile, pixsize[2], show)

    bgr_mean_slice = bgr_mean[:,int(cx-halfwidth[1]):int(cx+halfwidth[2])].T
    if show: plt.imshow(bgr_mean_slice); plt.show()
    profile = bgr_mean_slice.mean(axis=0)
    y_center = getCenter1D(profile, pixsize[1], show)
        
    return y_center, x_center
'''

def getBackground(data, pixsize, z_backgr, y_center, x_center, show):
    #plt.imshow(data.sum(axis=1)); plt.show()
    #print z_backgr

    mean_bgr = data[z_backgr].mean(axis=0)

    ry_big, rx_big = 90/pixsize[1:]
    ry_small, rx_small = 60/pixsize[1:]

    y,x = np.ogrid[-y_center:mean_bgr.shape[0]-y_center, -x_center:mean_bgr.shape[1]-x_center]
    mask = ((x/rx_big)**2 + (y/ry_big)**2 < 1) & ((x/rx_small)**2 + (y/ry_small)**2 > 1)

    bgr = data[z_backgr, mask]

    if show:
        maskPlot(mean_bgr, mask, overlay_type='area'); plt.show()

    return bgr.mean()

def getSphereAngles(data, pixsize, y_center, x_center, bgr_mean, show):
    mean_img = gaussian_filter(data, 5/pixsize).mean(axis=0)

    local_maxi = feature.peak_local_max(mean_img, min_distance=5, indices=True, num_peaks=2, threshold_abs=bgr_mean, num_peaks_per_label=1)
    mask = np.zeros_like(mean_img)
    mask[local_maxi.T[0],local_maxi.T[1]] = 1.0

    if show:
        maskPlot(mean_img, mask, overlay_type='area'); plt.show()

    activities = []
    for i, pk in enumerate(local_maxi):
        msk = getSphereMask2D(mean_img.shape, pk[0], pk[1], 20/pixsize[1], 20/pixsize[2])
        activities.append((mean_img[msk] - bgr_mean).sum())
    local_maxi = local_maxi[np.argsort(activities)].astype(float)

    sphere_angle = [np.arctan2(xy[0]-y_center, xy[1]-x_center) for xy in local_maxi]
    direction = np.sign(sphere_angle[1]-sphere_angle[0])

    sphere_angles = direction * np.arange(0, 6)*2*np.pi/6.0
    sphere_angles -= [sphere_angles[5]-sphere_angle[1]]
    
    #sphere_angles += -3. * np.pi/180.
    
    spheres_xy = np.array((np.sin(sphere_angles), np.cos(sphere_angles))).T * 57.2

    if show:
        mask = np.zeros_like(mean_img)
        for i in range(6):
            cy, cx = [y_center, x_center] + spheres_xy[i] / pixsize[1:]
            rz, ry, rx = 0.5 * SPHERES_MM[i] / pixsize
            mask[ getSphereMask2D(mean_img.shape, cy, cx, ry, rx) ] = 1
        maskPlot(mean_img, mask, overlay_type='area'); plt.show()

    return sphere_angles
    


def getSphereMask3D(shape, cz, cy, cx, rx, ry, rz):
    dz, dy, dx = shape
    z, y, x = np.ogrid[-cz:dz-cz, -cy:dy-cy, -cx:dx-cx]
    return ((x/rx)**2 + (y/ry)**2 + (z/rz)**2 <= 1)
'''
def getLungInsertMask(shape, cz, cy, cx, rx, ry):
    dz, dy, dx = shape
    y, x = np.ogrid[-cy:dy-cy, -cx:dx-cx]
    return np.tile(((x/rx)**2 + (y/ry)**2 <= 1), (dz, 1, 1))

def recoveryCoefficient(value, background, fillratio):
    if fillratio != 0:
        return (value/background - 1) / (fillratio - 1)
    else:
        return 1 - value/background
'''
def getInsertData(image, mask, cz):
    insert3d = image[mask]
    insert2d = image[cz, mask[cz]]
    data = {
        "3d min": insert3d.min(),
        "3d max": insert3d.max(),
        "3d mean": insert3d.mean(),
        "3d std": insert3d.std(),
        "3d sum": insert3d.sum(),
        "volume": mask.sum(),

        "2d min": insert2d.min(),
        "2d max": insert2d.max(),
        "2d mean": insert2d.mean(),
        "2d std": insert2d.std(),
        "2d sum": insert2d.sum(),
        "area": mask[cz].sum(),
    }
    return data

def getError(data, pixsize, sphere_angles, z_c, y_c, x_c, d_a):
    sph_coords = np.array((np.sin(sphere_angles+d_a), np.cos(sphere_angles+d_a))).T * 57.2
    spheres_yx = [y_c, x_c] + sph_coords / pixsize[1:]
    total_val = 0.0
    for i in range(6):
        index = i+1    
        cy, cx = spheres_yx[i]
        rz, ry, rx = 0.5 * SPHERES_MM[i] / pixsize
        mask = getSphereMask3D(data.shape, z_c, cy, cx, rx, ry, rz)
        total_val += data[np.where(mask)].mean()
    return -total_val
    
def minimizeByPos(p, data, pixsize, sphere_angles, z_c, d_a):
    y_c, x_c = p
    return getError(data, pixsize, sphere_angles, z_c, y_c, x_c, d_a)

def minimizeByAngle(p, data, pixsize, sphere_angles, y_c, x_c):
    z_c, d_a = p
    return getError(data, pixsize, sphere_angles, z_c, y_c, x_c, d_a)

def minimizeSingle(p, data, pixsize, rx, ry, rz):
    cz, cy, cx = p
    mask = getSphereMask3D(data.shape, cz, cy, cx, rx, ry, rz)
    return -data[np.where(mask)].mean()


'''
def analyzeSphere(sphere_index, insertdata, background, fillratio, pixsize):
    print ("%s %i 3D: %.2fml (%.2f), RC: %.2f, T:N ratio: %.2f - 2D: %.1fmm (%.1f), RC: %.2f, T:N ratio: %.2f"%(
        "Sphere", sphere_index+1, 
        insertdata["volume"] * np.prod(0.1*pixsize),
        SPHERES_ML[sphere_index],
        recoveryCoefficient(insertdata["3d mean"], background, fillratio),
        insertdata["3d mean"]/background,
        np.sqrt(insertdata["area"] * np.prod(pixsize[1:]) / np.pi)*2,
        SPHERES_MM[sphere_index],
        recoveryCoefficient(insertdata["2d mean"], background, fillratio),
        insertdata["2d mean"]/background,
    ))
    
def analyzeLungInsert(insertdata, background, pixsize):
    print ("%s 3D: RC: %.2f, T:N ratio: %.2f - 2D: %.1fmm (%.1f), RC: %.2f, T:N ratio: %.2f"%(
        "Lung Insert", 
        recoveryCoefficient(insertdata["3d mean"], background, 0),
        insertdata["3d mean"]/background,
        np.sqrt(insertdata["area"] * np.prod(pixsize[1:]) / np.pi)*2,
        LUNG_INSERT_RADIUS * 2,
        recoveryCoefficient(insertdata["2d mean"], background, 0),
        insertdata["2d mean"]/background,
    ))
'''


def analyze_suv(data, pixsize, margin_mm, show, savefigname):
    data = np.copy(data)
    otsu_thres = skimage.filters.threshold_otsu(data)
    
    mask = data > otsu_thres
    
    iterations = int(np.ceil(margin_mm/pixsize[1]))

    new = binary_opening(mask, iterations=2, structure=np.ones((3,3,3)))
    
    if iterations:
        mask = binary_erosion(mask, iterations=iterations)

    plt.figure(figsize=(12,6))
    plt.subplot(121); maskPlot(data.mean(axis=0), mask.max(axis=0), overlay_type='area')
    plt.subplot(122); maskPlot(data.mean(axis=1), mask.max(axis=1), overlay_type='area')
    plt.savefig(savefigname)

    slicesuv = []
    slicedev = []
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for z in range(len(data)):
            tmpsuv = data[z,:,:][mask[z,:,:]].mean()
            tmpdev = data[z,:,:][mask[z,:,:]].std()
            if tmpsuv > 0:
                slicesuv.append(tmpsuv)
                slicedev.append(tmpdev)

    return data[mask].mean(),slicesuv,slicedev


def analyze_iq(data, pixsize, zoomfactor, fit_individual, show,savefigname):
    data = np.copy(data)
    
    data = rotatePhantom(data, pixsize, show)
    
    data = cropPhantom(data, pixsize, show)
        
    z_spheres, z_backgr = getSphereAndBackgroundSlices(data, pixsize, show)
    
    y_center, x_center = np.array(data.shape)[1:]/2
    
    bgr_mean = getBackground(data, pixsize, z_backgr, y_center, x_center, show)
    
    sphere_angles = getSphereAngles(data, pixsize, y_center, x_center, bgr_mean, show)
    
    ry, rx = 100/pixsize[1:]
    data = data[z_spheres, int(y_center-ry):int(y_center+ry), int(x_center-rx):int(x_center+rx)]
    data = zoom(data, zoomfactor, order=0)
    pixsize /= zoomfactor

    z_c, y_c, x_c = np.array(data.shape)/2
    d_a = 0
    
    print ('Starting: ', z_c/zoomfactor, y_c/zoomfactor, x_c/zoomfactor, d_a)

    if not fit_individual:    
        args = (data, pixsize, sphere_angles, z_c, d_a)
        res = minimize(minimizeByPos, [y_c, x_c], args=args, method='Nelder-Mead')
        y_c, x_c = res.x[0], res.x[1]
        
        print ('Fitted pos 1: ', z_c/zoomfactor, y_c/zoomfactor, x_c/zoomfactor, d_a)
        
        args = (data, pixsize, sphere_angles, y_c, x_c)
        res = minimize(minimizeByAngle, [z_c, d_a], args=args, method='Nelder-Mead')
        z_c, d_a = res.x[0], res.x[1]
        
        print ('Fitted ang 1: ', z_c/zoomfactor, y_c/zoomfactor, x_c/zoomfactor, d_a)
        
        args = (data, pixsize, sphere_angles, z_c, d_a)
        res = minimize(minimizeByPos, [y_c, x_c], args=args, method='Nelder-Mead')
        y_c, x_c = res.x[0], res.x[1]
        
        print ('Fitted pos 2: ', z_c/zoomfactor, y_c/zoomfactor, x_c/zoomfactor, d_a)
        
        args = (data, pixsize, sphere_angles, y_c, x_c)
        res = minimize(minimizeByAngle, [z_c, d_a], args=args, method='Nelder-Mead')
        z_c, d_a = res.x[0], res.x[1]

        print ('Fitted ang 2: ', z_c/zoomfactor, y_c/zoomfactor, x_c/zoomfactor, d_a)
    

    sph_coords = np.array((np.sin(sphere_angles + d_a), np.cos(sphere_angles + d_a))).T * 57.2
    spheres_yx = [y_c, x_c] + sph_coords / pixsize[1:]

    labeled_mask = np.zeros_like(data, dtype=int)
    fitted = []
    means = []
    for i in range(6):
        cy, cx = spheres_yx[i]
        rz, ry, rx = 0.5 * SPHERES_MM[i] / pixsize
        
        if fit_individual: #Fit individual spheres
            args = (data, pixsize, rx, ry, rz)
            res = minimize(minimizeSingle, [z_c, cy, cx], args=args, method='Nelder-Mead')
            z_c, cy, cx = res.x[0], res.x[1], res.x[2]
            fitted.append(res.x)
        
        
        ### Actual spheres
        mask = getSphereMask3D(data.shape, z_c, cy, cx, rx, ry, rz)
        insertdata = getInsertData(data, mask, int(round(z_c)))
        labeled_mask[mask] = i+1
        
        means.append(insertdata["3d mean"])
        
    if show:
        maskPlot(data.max(axis=1), labeled_mask.max(axis=1), vmin=0, vmax=bgr_mean+0.5*(data.max()-bgr_mean)); plt.show()
        maskPlot(data.max(axis=0), labeled_mask.max(axis=0), vmin=0, vmax=bgr_mean+0.5*(data.max()-bgr_mean)); plt.show()
    
    if savefigname:
        maskPlot(data.max(axis=0), labeled_mask.max(axis=0), vmin=0, vmax=bgr_mean+0.5*(data.max()-bgr_mean))
        plt.savefig(savefigname)
    
    return bgr_mean, np.array(means)

