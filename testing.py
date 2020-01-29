import os
import pydicom as dicom
from wad_qc.modulelibs import wadwrapper_lib
import numpy as np
import matplotlib.pyplot as plt
import earliq_lib as earllib


def loadData(dcm_folder):
    data = []
    position = []
    for fn in os.listdir(dcm_folder):
        path = os.path.join(dcm_folder, fn)
        ds = dicom.read_file(path)
        data.append(ds.pixel_array.astype(np.float64) * ds.RescaleSlope)
        position.append(float(ds.ImagePositionPatient[2]))
    position = np.array(position)
    sorting = np.argsort(position)
    position = position[sorting]
    data = np.array(data)[sorting]
    pixsize = [float(x) for x in ds.PixelSpacing] + [(position[-1]-position[0])/(len(position)-1)]

    pixsize = np.abs(np.array(pixsize)[::-1])
    return data, pixsize


def earlsuv_analysis(dcm_folder):    
    ## TODO read these from config
    

    ## TODO read from manual input
        
    actual_concentration = 70E6/6250
    
    pixeldataIn, pixsize = loadData(dcm_folder)
    
    show = 1
    margin_mm = 10
    
    measured_concentration = earllib.analyze_suv(pixeldataIn, pixsize, margin_mm, show)
    print(measured_concentration)
    

def earliq_analysis(dcm_folder):    
    ## TODO read these from config
    spheres_ml = [26.42, 11.49, 5.57, 2.57, 1.15, 0.52]
    lung_ml = 194
    torso_empty_ml = 9700

    inserts_ml = sum(spheres_ml) + lung_ml
    background_ml = torso_empty_ml - inserts_ml

    ## TODO read from manual input
    sol1_con = 100. / 800. #MBq/ml
    bgr1_act = 50.0
    fill_ratio = sol1_con / bgr1_act * background_ml
    
    fill_ratio = 9700/1000
    
    for fit_individual in [0, 1]:
        show=1
        zoomfactor = 4.
        #fit_individual = 1
        pixeldataIn, pixsize = loadData(dcm_folder)
        
        bgr_mean, sphere_means = earllib.analyze(pixeldataIn, pixsize, zoomfactor, fit_individual, show)
        
        RCs = sphere_means/bgr_mean/fill_ratio
        print(RCs)
        
        SPHERES_MM = [10, 13, 17, 22, 28, 37]
        SPHERES_ML = [4/3.*np.pi*(0.1*d/2.)**3 for d in SPHERES_MM]
        plt.plot(SPHERES_ML, RCs)
    plt.show()
    

data = "Testdata/CALIBRATION QC/PETFDGSLACEARL20191114"
earlsuv_analysis(data)

exit()

data = "Testdata/IMAGE QC/PETFDGSLACEARLIMAGEQC20191114"
earliq_analysis(data)
