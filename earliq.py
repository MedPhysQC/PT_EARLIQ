#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
#
# Changelog:
#   20190426: initial version

from __future__ import print_function

__version__ = '20190628'
__author__ = 'rvrooij,ddickerscheid,tdwit'


import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import pydicom_series as dcms

from wad_qc.modulelibs import wadwrapper_lib


import earliq_lib as earllib
import numpy as  np
from skimage import filters 
from scipy.ndimage.interpolation import zoom
import scipy.misc
from skimage.transform import radon
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate
from PIL import Image
from skimage.filters import threshold_otsu

import matplotlib
#matplotlib.use('tkagg') # Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt


if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 


try:
    import pydicom as dicom
except ImportError:
    import dicom


def logTag():
    return "[earliq] "

    
def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database
    Workflow:
        1. Read only headers
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt) 

def earliq_analysis(data, results, config):
    """
    Read selected dicomfields and write to IQC database
    Workflow:
        1. Run tests
    """
    try:
        params = config["actions"]["earliq"]['params']
    except KeyError:
        params = {} # Params will contain the phantom specific parameters 

    try:
        info = params.find("info").text
    except AttributeError:
        info = 'qc' # selected subset of DICOM headers informative for QC testing


    try:
        manualinput = config['manual_input']
    except KeyError:
        manualinput = {}


    print('manualinput:')

    for key in manualinput.keys():
        print(manualinput[key])

    
    ## TODO read these from config
    spheres_ml =  [float(x) for x in params["spheres_ml"].split(',')]  #[26.42, 11.49, 5.57, 2.57, 1.15, 0.52]
    lung_ml = float(params["lung_ml"]) #194
    torso_empty_ml = float(params["torso_empty_ml"]) # 9700

    inserts_ml = sum(spheres_ml) + lung_ml
    background_ml = torso_empty_ml - inserts_ml

    half_life_secs = float(params["half_life_secs"]) #TO DO: get the halflife/radiopharmaceutical info from the dicom header -> move to manual input
    
    ## TODO read from manual input
    sol1_con = 100. / 800. #MBq/ml
    bgr1_act = 50.0

    fill_ratio = sol1_con / bgr1_act * background_ml




    

    # Load data
    inputseries =   data.getAllSeries()[0]
    print(float(inputseries[0].PixelSpacing[0]))

    dcmInfile, pixeldataIn, dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], headers_only=False, logTag=logTag())

    print('-----' ,np.shape(pixeldataIn))
    plt.imshow(pixeldataIn.max(axis=0))
    plt.show()
    ## 2. Check data format
     
    
    ### Rotate the data such that the spheres in the phantom are exactly in plane.
    # The angles are found by thresholding the phantom body and taking the radon transform along every axis. 
    # This method assumes certain symmetries and straight edges in the phantom. 

    rotdata = pixeldataIn
    for axis in (1,2,0):
        rotdata = earllib.correctAxis(rotdata, axis, show=False)

        
    #im = Image.fromarray(rotdata.max(axis=1)[:,:45]).convert('RGB')
    #name = 'rotateddata'
    #fn = '%s.jpg'%name
    #im.save(fn)
    #results.addObject(name, fn)
    
    plt.imshow(rotdata.max(axis=1)[:,:45]); plt.show()
    bgr_threshold = threshold_otsu(rotdata.max(axis=1)[:,:45])
    
    #plt.title("ROTDATA")
    earllib.myPlot(rotdata.max(axis=2),  (rotdata > bgr_threshold).max(axis=2));
    plt.show()    


    zoomfactor = 4.0
    sphere_index = earllib.getSphereIndex(pixeldataIn)
    print('sphereindex',sphere_index)
    #sphere_index = 33
    
    pixsize = float(inputseries[0].PixelSpacing[0])
    print('pixsize',pixsize)
    
    width_of_sphere_slice = 30. #mm
    width = int(np.ceil( width_of_sphere_slice / pixsize ))

    z_spheres = np.s_[sphere_index-width:sphere_index+width]
    z_backgr = np.s_[sphere_index+width:sphere_index+3*width]

    print('z_spheres',z_spheres)
    
    x_center, y_center = earllib.getCenter(rotdata[z_spheres], sigma=8/pixsize, show=True)
    radius = int(100/pixsize)
    chop = rotdata[z_spheres, y_center-radius:y_center+radius, x_center-radius:x_center+radius]

    zoomed = zoom(chop, zoomfactor, order=0)
    pixsize /= zoomfactor


    image = zoomed.max(axis=2)
    #plt.imshow(image)
    
    '''
    im = Image.fromarray(image).convert('RGB')
    name = 'overview1'
    fn = '%s.jpg'%name
    im.save(fn)
    results.addObject(name, fn)
    '''

    image = zoomed.max(axis=0)
    #plt.imshow(image)
    '''
    im = Image.fromarray(image).convert('RGB')
    name = 'overview2'
    fn = '%s.jpg'%name
    im.save(fn)
    results.addObject(name, fn)
    '''
    

    ### FIXME: background value should be obtained using NEMA standard.

    bgrmask = np.zeros_like(rotdata[z_backgr],dtype=bool)
    bgrmask[:,60:90,40:50] = True #TO  DO in config?
    

    bgr = rotdata[z_backgr][bgrmask]
    
    bgr_mean = bgr.mean()
    print ('background max:',bgr.max(), '\n background mean:',bgr.mean(), '\n background std:', bgr.std())

    SPHERE_CENTER_R = 57.2 / pixsize
    SPHERE_ANGLES = np.arange(0, 6)*2*np.pi/6.0
    SPHERE_COORDS = np.array((np.cos(SPHERE_ANGLES), np.sin(SPHERE_ANGLES))).T * SPHERE_CENTER_R
    SPHERE_RADIUS = np.array((18.5, 14.0, 11.0, 8.5, 6.5, 5.0)) / pixsize

    #Start at smallest sphere:
    SPHERE_COORDS = SPHERE_COORDS[::-1]
    SPHERE_RADIUS = SPHERE_RADIUS[::-1]
    
    dz, dy, dx = zoomed.shape

    cz = earllib.getSphereIndex(gaussian_filter(zoomed, 2*zoomfactor))
    print('getSphereIndex',cz)
    x_center, y_center = earllib.getCenter(zoomed, sigma=8/pixsize, show=True)
    
    vols, means, maxs, RCs = [],[],[],[]

    text = "%s %i, volume: %.2f ml, RC: %.2f, mean/bgr: %.2f, max/bgr: %.2f"
    mask = np.zeros_like(zoomed, dtype=int)
    for i in range(6):
        cx, cy = SPHERE_COORDS[i] + [x_center, y_center]
        z, y, x = np.ogrid[-cz:dz-cz, -cy:dy-cy, -cx:dx-cx]
        rsq = SPHERE_RADIUS[i]**2
        index = i+1
    
        thismask = (x*x + y*y + z*z <= rsq)
    
        mask[thismask] = index
    
        meanval = zoomed[thismask].mean()
        maxval = zoomed[thismask].max()
        volume = thismask.sum() * (0.1*pixsize)**3
        RC = meanval/bgr_mean / fill_ratio
    
        vols.append(volume)
        means.append(meanval/bgr_mean)
        maxs.append(maxval/bgr_mean)
        RCs.append(RC)
    
        print (text%("Sphere", index, volume, RC, meanval/bgr_mean, maxval/bgr_mean))

    cx, cy = x_center, y_center
    y, x = np.ogrid[-cy:dy-cy, -cx:dx-cx]
    rsq = (22.25 / pixsize)**2
    index = 7
    mask[:, x*x + y*y <= rsq] = index
    meanval = zoomed[mask==index].mean()
    maxval = zoomed[mask==index].max()
    volume = (mask==index).sum() * (0.1*pixsize)**3
    RC = meanval/bgr_mean / fill_ratio
    
    #print text%("Insert", index, volume, RC, meanval/bgr_mean, maxval/bgr_mean)
    #Add these plots to results
    #earllib.maskPlot(zoomed.max(axis=1), mask.max(axis=1))
    #earllib.maskPlot(zoomed.max(axis=0), mask.max(axis=0), vmin=bgr_mean, vmax=(3*bgr_mean + maxval)/4.0); plt.show()
    #earllib.maskPlot(zoomed.max(axis=0), mask.max(axis=0), vmin=2*bgr_mean); plt.show()

    im = Image.fromarray(zoomed.max(axis=1)).convert('RGB')
    name = 'testdd.png'
    fn = '%s.jpg'%name
    im.save(fn)
    results.addObject(name, fn)

    #plt.plot(vols, means, 'o', vols, maxs, 'o', [0, max(vols)],[fill_ratio, fill_ratio]); plt.show()
    #plt.plot(vols, RCs); plt.ylim([0,1]); plt.show()
    
    ##  Add results to'result' object
    varname = 'pluginversion'
    #results.addString(varname, varval)
    
    results.addFloat('rc',1.0)



if __name__ == "__main__":
    data, results, config = pyWADinput()

    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'header_series':
            header_series(data, results, action)
        
        elif name == 'earliq':
            earliq_analysis(data, results, config)


    results.write()
