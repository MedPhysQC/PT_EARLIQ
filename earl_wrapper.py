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

__version__ = '20200201'
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
        info = params["info"]
    except AttributeError:
        info = 'qc' # selected subset of DICOM headers informative for QC testing


    try:
        manualinput = config['manual_input']
    except KeyError:
        manualinput = {}


    # Load data
    inputseries =   data.series_filelist[0]
    pixeldataIn,pixsize = earllib.loadData(inputseries)

    header = dicom.read_file(inputseries[0],stop_before_pixels=True)
    
    
    ## Phantom parameters from config
    spheres_ml =  [float(x) for x in params["spheres_ml"].split(',')]  #[26.42, 11.49, 5.57, 2.57, 1.15, 0.52]
    phantom_bg_vol = float(params["phantom_bg_vol"]) # 9700

    
    ## Manual input parameters:8042/app/explorer.html#instance?uuid=42e9c86d-d7325651-ba3a40aa-08bf3871-1e768553

    #1 READ parameters
    stock_vol = float(manualinput["stock_vol"]['val'])

    datetimeformat = "%Y-%m-%dT%H:%M:%S"

    acq_datetime = datetime.strptime(manualinput['acq_date']['val']+"T"+manualinput['acq_time']['val'],datetimeformat)
    
    print(manualinput["dose_bg"])
    dose_bg =  float(manualinput["dose_bg"]['val'])
    dose_bg_datetime = datetime.strptime( manualinput['dose_bg_date']['val']+"T"+manualinput['dose_bg_time']['val'],datetimeformat)
 
    residual_dose_bg = float(manualinput[ "residual_dose_bg"]["val"])
    residual_dose_bg_datetime =  datetime.strptime( manualinput['residual_dose_bg_date']['val']+"T"+manualinput['residual_dose_bg_time']['val'],datetimeformat)
    
    dose_spheres = float(manualinput[ "dose_spheres"]['val'])
    dose_spheres_datetime = datetime.strptime( manualinput['dose_spheres_date']['val']+"T"+manualinput['dose_spheres_time']['val'],datetimeformat)
    
    residual_dose_spheres = float(manualinput["residual_dose_spheres"]['val'])
    residual_dose_spheres_datetime =  datetime.strptime( manualinput['residual_dose_spheres_date']['val']+"T"+manualinput['residual_dose_spheres_time']['val'],datetimeformat)

    #2 Calculate net background and stock activities
    
    print('Radiopharmaceutical Info')

    ris = header.RadiopharmaceuticalInformationSequence[0]
    half_life_secs = ris.RadionuclideHalfLife
    isotope = ris.RadionuclideCodeSequence[0].CodeMeaning

    results.addString('Isotope',isotope)
    results.addFloat('Half life',half_life_secs)

    #tref  = dose_bg_datetime
    tref  = acq_datetime
    
    bgd = earllib.Activity(dose_bg,dose_bg_datetime,half_life_secs)
    bgr = earllib.Activity(residual_dose_bg,residual_dose_bg_datetime,half_life_secs)

    sd = earllib.Activity(dose_spheres,dose_spheres_datetime,half_life_secs)
    sr = earllib.Activity(residual_dose_spheres,residual_dose_spheres_datetime,half_life_secs)

    netbg = bgd.At(tref) - bgr.At(tref)
    netspheres = sd.At(tref) - sr.At(tref)
    
    sol1_con = netspheres / stock_vol #MBq/ml
    bgr1_con = netbg / phantom_bg_vol

    fill_ratio = sol1_con / bgr1_con

    print (fill_ratio)
    

    
    show=1
    zoomfactor = 4.
    fit_individual = 0 #This determines if the spheres should be fitted individually 
        
    bgr_mean, sphere_means = earllib.analyze_iq(pixeldataIn, pixsize, zoomfactor, fit_individual, show,'sphere registration')
        
    RCs = sphere_means/bgr_mean/fill_ratio
    SPHERES_MM = [10, 13, 17, 22, 28, 37]
    SPHERES_ML = [4/3.*np.pi*(0.1*d/2.)**3 for d in SPHERES_MM]

    EARL_RC_MIN_VOIA50 = [0.27,0.44,0.57,0.63,0.72,0.76]
    EARL_RC_MAX_VOIA50 = [0.43,0.60,0.73,0.78,0.85,0.89]

    
    for i in range(6):
        results.addFloat('Sphere '+str(SPHERES_MM[i])+' mm',RCs[i])

    
    filename = 'RCcurve.png'
    plt.figure(figsize=(12,6))
    plt.grid()
    plt.xlabel("Sphere diameter (mm)")
    plt.ylabel("Recovery coefficient")
    
    plt.plot(SPHERES_MM, RCs, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)

    plt.plot(SPHERES_MM,EARL_RC_MAX_VOIA50 , color='black', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12)


    plt.plot(SPHERES_MM,EARL_RC_MIN_VOIA50 , color='black', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12)

    
    plt.savefig(filename)
    results.addObject('RCcurve',filename)
    





def earlsuv_analysis(data, results, config):
    """
    Read selected dicomfields and write to IQC database
    Workflow:
        1. Run tests
    """


    # Load data
    inputseries =   data.series_filelist[0]
    pixeldataIn,pixsize = earllib.loadData(inputseries)
    header = dicom.read_file(inputseries[0],stop_before_pixels=True)

    
    try:
        params = config["actions"]["earlsuv"]['params']
    except KeyError:
        params = {} # Params will contain the phantom specific parameters 

    #try:
    #    info = params["info"] 
    #except AttributeError:
    #    info = 'qc' # selected subset of DICOM headers informative for QC testing

    if True:
        ge68suv = params['ge68suv']
    #except:
    #    ge68suv = False

    print ('------',ge68suv)
        
    if ge68suv == 'True':
        manualinput = {}
        
        datetimeformat = "%Y%m%dT%H%M%S"
        acq_datetime = datetime.strptime(header.AcquisitionDate+"T"+header.AcquisitionTime[0:6],datetimeformat)
    
        dose_bg = float(params['ge68activity'])
        dose_bg_datetime = datetime.strptime(params['calibrationdate']+"T"+"000000",datetimeformat)
        residual_dose_bg = 0.0
        residual_dose_bg_datetime = dose_bg_datetime
        
    else:    
        try:
            manualinput = config['manual_input']
        except KeyError:
            manualinput = {}


            ## Manual input parameters:8042/app/explorer.html#instance?uuid=42e9c86d-d7325651-ba3a40aa-08bf3871-1e768553

        #1 READ parameters
        datetimeformat = "%Y-%m-%dT%H:%M:%S"
        acq_datetime = datetime.strptime(manualinput['acq_date']['val']+"T"+manualinput['acq_time']['val'],datetimeformat)
    
        print(manualinput)
    
        dose_bg =  float(manualinput["dose_bg"]["val"])
        #dose_bg = 71.0
        dose_bg_datetime =  datetime.strptime( manualinput['dose_bg_date']['val']+"T"+manualinput['dose_bg_time']['val'],datetimeformat)
        #dose_bg_datetime = datetime.strptime('2019-11-14T16:30',datetimeformat)
        residual_dose_bg =  float(manualinput[ "residual_dose_bg"]["val"])
        #residual_dose_bg = 0.0
        residual_dose_bg_datetime =  datetime.strptime( manualinput['residual_dose_bg_date']['val']+"T"+manualinput['residual_dose_bg_time']['val'],datetimeformat)
        #residual_dose_bg = datetime.strptime('2019-11-14T16:44',datetimeformat) 
    

        results.addFloat('Phantom dose',dose_bg)
        results.addDateTime('Phantom dose DateTime', dose_bg_datetime) 

        results.addFloat('Residual dose',residual_dose_bg)
        results.addDateTime('Residual dose DateTime', residual_dose_bg_datetime) 

    

    #2 Calculate net background and stock activities
        
    print('Radiopharmaceutical Info')
    ris = header.RadiopharmaceuticalInformationSequence[0]
    half_life_secs = ris.RadionuclideHalfLife
    isotope = ris.RadionuclideCodeSequence[0].CodeMeaning
    
    results.addString('Isotope',isotope)
    results.addFloat('Half life',half_life_secs)
    
    studydate = str(header.StudyDate)
    studytime = str(header.StudyTime)

    scan_datetime = datetime.strptime(studydate+studytime,"%Y%m%d%H%M%S")
    
    if header.Units != 'BQML':
        raise ValueError('A very specific bad thing happened: units are not BQML!')
    
    ## Phantom parameters from config
    phantom_vol = float(params["phantom_vol"]) # 9700
    results.addFloat('PhantomVolume',phantom_vol)
    

    #3 Calculate SUV

    #tref  = dose_bg_datetime
    tref = acq_datetime

    print (half_life_secs,dose_bg,dose_bg_datetime)
    bgd = earllib.Activity(dose_bg,dose_bg_datetime,half_life_secs)
    bgr = earllib.Activity(residual_dose_bg,residual_dose_bg_datetime,half_life_secs)

    
    #bgd = earllib.Activity(dose_bg,scan_datetime,half_life_secs)
    #bgr = earllib.Activity(residual_dose_bg,scan_datetime,half_life_secs)

    netbg = bgd.At(tref) - bgr.At(tref)
    admincon = netbg / phantom_vol   #MBQ/ml 

    
    show=0
    marginmm = 1.
    results.addFloat('Margin',marginmm)

    filename = 'SUVfit.png'

    measuredmean,slicecountlist, slicesdevlist = earllib.analyze_suv(pixeldataIn, pixsize, marginmm, show,filename)
    measuredmean = measuredmean/np.power(10,6) #Divide by 10^6 to get Bq/ml

    results.addObject('SUV mask',filename)

    plt.figure(figsize=(12,6))
    plt.grid()
    slicesuvs = [ (x/np.power(10,6))/admincon for x in slicecountlist]
    plt.plot(slicesuvs)
    plt.savefig('SUVslice.png')
    results.addObject('SUV/slice','SUVslice.png')


    plt.figure(figsize=(12,6))
    plt.grid()
    plt.plot(slicesdevlist)
    plt.savefig('SUVstdevslice.png')
    results.addObject('SUVstdev/slice','SUVstdevslice.png')


    
    results.addFloat('Administered concentration',admincon)
    results.addFloat('Measured concentration',admincon)
    results.addFloat('SUV',measuredmean/admincon)

    
      


    

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
