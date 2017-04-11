#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for 3D ptychography."""

import dxchange
import tomopy
import xraylib as xl
import numpy as np
import scipy as sp
import pyfftw


PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


def wavelength(energy):
    """Calculates the wavelength [cm] given energy [keV].
    
    Parameters
    ----------
    energy : scalar

    Returns
    -------
    scalar
    """
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


class Material(object):
    """Material property definitions.

    Attributes
    ----------
    compound : string
        Molecular formula of the material.
    density : scalar
        Density of the compound [g/cm^3].
    energy : scalar
        Illumination energy [keV].
    """

    def __init__(self, compound, density, energy):
        self.compound = compound
        self.density = density
        self.energy = energy

    @property
    def beta(self):
        """Absorption coefficient."""
        return xl.Refractive_Index_Im(self.compound, self.energy, self.density)

    @property
    def delta(self):
        """Decrement of refractive index."""
        return 1 - xl.Refractive_Index_Re(self.compound, self.energy, self.density)

    @property
    def wavelength(self):
        """Wavelength of illumination [cm]."""
        return wavelength(energy)


class Object(object):
    """Discrete object represented in a 3D regular grid.
    
    Attributes
    ----------
    material : object
        Material of the object.
    grid : ndarray
        3D regular grid with binary values defining spatial
        distribution of the object in space.
    voxelsize : scalar [cm]
        Size of the voxels in the grid.
        
    """
    def __init__(self, material, grid, voxelsize):
        self.beta = material.beta * grid
        self.delta = material.delta * grid
        self.energy = material.energy
        self.voxelsize = voxelsize


class Probe(object):
    """Illumination probe represented on a 2D regular grid.

    A finite-extent circular shaped probe is represented as 
    a complex wave. The intensity of the probe is maximum at 
    the center and damps to zero at the borders of the frame. 
    
    Attributes
    ----------
    size : int
        Size of the square 2D frame for the probe.
    damp : float
        Value between 0 and 1 determining where the 
        dampening of the intensity will start.
    maxint : float
        Maximum intensity of the probe at the center.
    """

    def __init__(self, size, damp, maxint):
        self.size = size
        self.damp = damp
        self.maxint = maxint

    @property
    def __weights(self):
        """Returns a circular mask with weights from 0 to 1."""
        r, c = np.mgrid[:self.size, :self.size] + 0.5
        rad = np.sqrt((r - self.size/2)**2 + (c - self.size/2)**2)
        img = np.zeros((self.size, self.size))
        rmin = np.sqrt(2) * 0.5 * self.damp * rad.max()
        rmax = np.sqrt(2) * 0.5 * rad.max()
        zone = np.logical_and(rad > rmin, rad < rmax)
        img[rad < rmin] = 1.0
        img[rad > rmax] = 0.0
        img[zone] = (rmax - rad[zone]) / (rmax - rmin)
        return img

    @property
    def amplitude(self):
        """Amplitude of the probe wave"""
        return np.sqrt(self.maxint) * self.__weights

    @property
    def phase(self):
        """Phase of the complex probe wave."""
        return 0.0 * self.__weights


class Detector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def raster_scan(init, dx, dy, dt, nx, ny, nt):
    """Calculates raster-scan coordinates.

    Parameters
    ----------
    init : dict
        Initial state of the [x, y, theta] geometry.
    dx : int
        Step size on x-axis.
    dy : int
        Step size on y-axis.
    dt : float
        Step size on theta-axis.
    nx : int
        Number pf steps on x-axis.
    ny : int
        Number pf steps on y-axis.
    nt : float
        Number pf steps on theta-axis.

    Returns
    -------
    cycler object
    """
    from cycler import cycler
    a = cycler(x=np.arange(init['x'], nx, dx))
    b = cycler(y=np.arange(init['y'], ny, dy))
    c = cycler(theta=np.arange(init['theta'], nt, dt))
    return a * b * c


def radon(obj, ang):
    pb = tomopy.project(obj.beta, ang, pad=False) * obj.voxelsize
    pd = tomopy.project(obj.delta, ang, pad=False) * obj.voxelsize
    return np.exp(1j * (2 * PI) / wavelength(obj.energy) * (pd + 1j * pb))



# Load a 3D object.
grid = dxchange.read_tiff('/home/beams/DGURSOY/Data/Ptycho/obj.tiff')

# Material property.
mat = Material('Au', 17.31, 5)

# Creat object.
obj = Object(mat, grid, 1e-7)

# Create probe.
prb = Probe(32, 0.7, 1000.0)

# Define scan geometry.
geo = raster_scan(
    {'x' : 0, 'y' : 0, 'theta' : 0}, 
    1, 1, 1, 
    1, 1, 180)

for g in geo:
    print (g)

# Calculate radon transform.
# rad = radon(obj, geo)

import matplotlib.pyplot as plt
plt.imshow(prb.amplitude, interpolation='none')
plt.show()

