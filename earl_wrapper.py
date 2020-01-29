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
from datetime import datetime

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


    # Load data
    inputseries =   data.getAllSeries()[0]
    pixelDataImn,pixsize = earllib.loadData(inputseries)




    
    
    ## Phantom parameters from config
    spheres_ml =  [float(x) for x in params["spheres_ml"].split(',')]  #[26.42, 11.49, 5.57, 2.57, 1.15, 0.52]
    phantom_bg_vol = float(params["phantom_bg_vol"]) # 9700
    stock_vol = float(params["stock_vol"])

    
    ## Manual input parameters

    #1 READ parameters
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    
    print(manualinput["dose_bg"])
    dose_bg = float(manualinput["dose_bg"])
    dose_bg_datetime = datetime.strptime( manualinput["dose_bg_date"] +' '+ manualinput["dose_bg_time"],datetimeformat)
 
    residual_dose_bg = float(manualinput[ "residual_dose_bg"])
    residual_dose_bg_datetime =   datetime.strptime(manualinput["residual_dose_bg_date"] +' '+ manualinput["residual_dose_bg_time"],datetimeformat)

    
    dose_spheres =  float(manualinput[ "dose_spheres"])
    dose_spheres_datetime =  datetime.strptime(manualinput[ "dose_spheres_date"]+' '+manualinput[ "dose_spheres_time"] ,datetimeformat)

    
    residual_dose_spheres = float(manualinput["residual_dose_spheres"])
    residual_dose_spheres_datetime =  datetime.strptime(manualinput[ "residual_dose_spheres_date"]+' '+manualinput[ "residual_dose_spheres_time"],datetimeformat)

    #2 Calculate net background and stock activities
    print(float(inputseries[0].PixelSpacing[0]))
    print('Radiopharmaceutical Info')
    ris = inputseries[0].RadiopharmaceuticalInformationSequence[0]
    half_life_secs = ris.RadionuclideHalfLife
    isotope = ris.RadionuclideCodeSequence[0].CodeMeaning
    print(isotope,half_life_secs)

    tref  = dose_bg_datetime
    
    bgd = earllib.Activity(dose_bg,dose_bg_datetime,half_life_secs)
    bgr = earllib.Activity(residual_dose_bg,residual_dose_bg_datetime,half_life_secs)

    sd = earllib.Activity(dose_spheres,dose_spheres_datetime,half_life_secs)
    sr = earllib.Activity(residual_dose_spheres,residual_dose_spheres_datetime,half_life_secs)

    netbg = bgd - bgr
    netspheres = sd - sr
    
    sol1_con = netspheres.At(tref) / stock_vol #MBq/ml
    bgr1_con = netbg.At(tref) / phantom_bg_vol

    fill_ratio = sol1_con / bgr1_con

    print (fill_ratio)
    

    
    for fit_individual in [0, 1]:
        show=1
        zoomfactor = 4.
        #fit_individual = 1
        pixeldataIn, pixsize = loadData(dcm_folder)
        
        bgr_mean, sphere_means = earllib.analyze_iq(pixeldataIn, pixsize, zoomfactor, fit_individual, show)
        
        RCs = sphere_means/bgr_mean/fill_ratio
        print(RCs)
        
        SPHERES_MM = [10, 13, 17, 22, 28, 37]
        SPHERES_ML = [4/3.*np.pi*(0.1*d/2.)**3 for d in SPHERES_MM]
        plt.plot(SPHERES_ML, RCs)
    plt.show()
    









    
    
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

        elif name == 'earlsuv':
            earlsuv_analysis(data, results, config)

            
    results.write()
