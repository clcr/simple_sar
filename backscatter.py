import os, sys
from osgeo import gdal
import numpy as np
from gdalconst import GA_ReadOnly
from scipy.ndimage.filters import uniform_filter
import matplotlib.pyplot as plt

def open_image_as_array(image_path, data_type):
    image_driver = gdal.Open(image_path, GA_ReadOnly)
    return image_driver.ReadAsArray().astype(data_type)


def save_array_as_image(array, out_path):
    image_driver = gdal.GetDriverByName('GTiff')
    output_image_driver = image_driver.Create(out_path, array.shape[1] +1, array.shape[0] + 1, 1, gdal.GDT_Float32)
    output_image_driver.GetRasterBand(1).WriteArray(array)
    output_image_driver = None


def filter_complex_image(image, size, mode='constant'):
    """Boxcar average of complex image. Created by splitting into real and imaginary
    components, boxcaring, and recombining. Assumes complex number are of the form (r + i)"""
    real = np.real(image)
    imag = np.imag(image)
    out = uniform_filter(real, size, mode=mode) + 1j*uniform_filter(imag, size, mode=mode)
    return out


os.chdir(os.path.dirname(__file__))

#Setup of variables
FILTER_SIZE = [11,21]

# Open input images (HH, HV and VV) in slant range geometry:
HH_m = open_image_as_array('data/master/afrisar_dlr_T2-0_SLC_HH.tiff', np.complex_)
HV_m = open_image_as_array('data/master/afrisar_dlr_T2-0_SLC_HV.tiff', np.complex_)
VV_m = open_image_as_array('data/master/afrisar_dlr_T2-0_SLC_VV.tiff', np.complex_)

HH_s = open_image_as_array('data/slave/afrisar_dlr_T2-5_SLC_HH.tiff', np.complex_)
HV_s = open_image_as_array('data/slave/afrisar_dlr_T2-5_SLC_HV.tiff', np.complex_)
VV_s = open_image_as_array('data/slave/afrisar_dlr_T2-5_SLC_VV.tiff', np.complex_)

fe = open_image_as_array(r'data/FE.tiff', np.complex_)

# Computing backscatter sigma in dB for all the polarisation channels:
HH_dB = 20 * np.log10(np.absolute(HH_m))
HV_dB = 20 * np.log10(np.absolute(HV_m))
VV_dB = 20 * np.log10(np.absolute(VV_m))

# Flat earth removal
HV_s_fe_rmvd = HV_s * (np.exp(fe*1j))

# Coherency
conj_HV_s_fe_rmvd = np.conj(HV_s_fe_rmvd)
m1 = HV_m * conj_HV_s_fe_rmvd

# Filtering (32x32 moving average. Assuming absolute of complex numbers)
m2 = filter_complex_image(m1, size=FILTER_SIZE, mode='constant')
HV_m = np.absolute(HV_m)**2
HV_s_fe_rmvd = np.absolute(HV_s_fe_rmvd)**2
m3 = uniform_filter(HV_m, size=FILTER_SIZE, mode='constant')
m4 = uniform_filter(HV_s_fe_rmvd, size=FILTER_SIZE, mode='constant')
f = np.sqrt(m3*m4) # This leads to NaNs at the boundaries
coherencey = m2 / f
interferogram = np.arctan(np.angle(coherencey))
magnitude = np.absolute(coherencey)

# Comparison
plt.figure()
plt.title("Backscatter (HH)")
plt.imshow(HH_dB)
plt.colorbar()

fig, (int_plot, mag_plot) = plt.subplots(1, 2)
int_im = int_plot.imshow(interferogram)
int_plot.set_title("Interferogram")
mag_im = mag_plot.imshow(magnitude)
mag_plot.set_title("Magnitude")
fig.colorbar(int_im, ax=int_plot)
fig.colorbar(mag_im, ax=mag_plot)
