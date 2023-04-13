import os
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from lmfit import Model

yEmpty = 2361
xEmpty = 5222
widthEmpty = 50

yGal = 2110
xGal = 4983
widthGal = 50

filters = ['g', 'i', 'r', 'y', 'z']

yStars = np.array([2881, 2759, 3268, 1563, 1767, 4199])
xStars = np.array([5617, 5608, 4399, 4057, 4199, 3590])
widthStar = 30


def getGalaxyImage(filter, xGal, yGal, widthGal, xEmpty, yEmpty, widthEmpty):
	hdul = fits.open('PS1/rings.v3.skycell.1803.087.stk.' + filter + '.unconv.fits')
	
	galaxy = hdul[1].data
	header = hdul[1].header
	EXPTIME = header['EXPTIME']
	GAIN = header['HIERARCH CELL.GAIN']
	NCOMBINE = 1
	
	hdul.close()
	
	# вырезаем галактику и пустую область рядом, чтобы оценить фон 
	# и прибавить его, симулиря реальный снимок
	
	galaxyImage = galaxy[xGal - widthGal: xGal + widthGal, yGal - widthGal: yGal + widthGal]
	emptyField = galaxy[xEmpty - widthEmpty: xEmpty + widthEmpty, yEmpty - widthEmpty: yEmpty + widthEmpty]
	
	sigma = np.std(emptyField)
	background = GAIN * sigma**2
	print(background)
	
	galaxyImage = galaxyImage + background
	
	plt.imshow(galaxyImage)
	plt.show()
	
	params = EXPTIME, GAIN, NCOMBINE, background
	return galaxyImage, params
	
def formPSF(filter, xStars, yStars, widthStar):
	hdul = fits.open('PS1/rings.v3.skycell.1803.087.stk.' + filter + '.unconv.fits')
	galaxy = hdul[1].data
	hdul.close()

	stars = np.zeros((2*widthStar+1, 2*widthStar+1, len(xStars)),)
	for i in range(len(xStars)):
		stars[:, :, i] = galaxy[xStars[i] - widthStar: xStars[i] + widthStar + 1, yStars[i] - widthStar: yStars[i] + widthStar + 1]
		empty_primary = fits.PrimaryHDU()
		image = fits.ImageHDU(stars[:, :, i])
	
		hdulStar = fits.HDUList([empty_primary, image])
		hdulStar.writeto('clean_images/' + filter + str(6 + i) + '.fits')
	
def formNewFits(filter, xGal, yGal, widthGal, xEmpty, yEmpty, widthEmpty):
	galaxyImage, params = getGalaxyImage(filter, xGal, yGal, widthGal, xEmpty, yEmpty, widthEmpty)
	EXPTIME, GAIN, NCOMBINE, background = params
	header = fits.Header()
	header['EXPTIME'] = EXPTIME
	header['GAIN'] = GAIN
	header['NCOMBINE'] = NCOMBINE
	
	empty_primary = fits.PrimaryHDU(header = header)
	image = fits.ImageHDU(galaxyImage, header = header)
	
	hdul = fits.HDUList([empty_primary, image])
	hdul.writeto('clean_images/' + filter + '.fits')
	
	return background

'''
backgroundG = formNewFits(filters[0], xGal, yGal, widthGal, xEmpty, yEmpty, widthEmpty)	
backgroundI = formNewFits(filters[1], xGal, yGal, widthGal, xEmpty, yEmpty, widthEmpty)	
backgroundR = formNewFits(filters[2], xGal, yGal, widthGal, xEmpty, yEmpty, widthEmpty)	
backgroundY = formNewFits(filters[3], xGal, yGal, widthGal, xEmpty, yEmpty, widthEmpty)	
backgroundZ = formNewFits(filters[4], xGal, yGal, widthGal, xEmpty, yEmpty, widthEmpty)	
print(backgroundG, backgroundI, backgroundR, backgroundY, backgroundZ)
'''

def displayFitResults(filter):
	hdul = fits.open('image_fitting/' + filter + 'NoPSF.fits')
	data = hdul[1].data
	bestFit = hdul[2].data
	residuals = hdul[3].data
	hdul.close()
	
	fig, axes = plt.subplots(1, 3)
	im1 = axes[0].imshow(data)
	plt.colorbar(im1, ax = axes[0])
	im2 = axes[1].imshow(bestFit)
	plt.colorbar(im2, ax = axes[1])
	im3 = axes[2].imshow(residuals)
	plt.colorbar(im3, ax = axes[2])
	
	axes[1].set_title('Filter ' + filter)
	
	plt.show()

'''	
displayFitResults('g')
displayFitResults('r')
displayFitResults('i')
displayFitResults('y')
displayFitResults('z')
'''

'''
formPSF('g', xStars, yStars, widthStar)
formPSF('r', xStars, yStars, widthStar)
formPSF('i', xStars, yStars, widthStar)
formPSF('y', xStars, yStars, widthStar)
formPSF('z', xStars, yStars, widthStar)
'''

FWHM = [9.93, 10.02, 9.38, 6.52, 7.73]

def generatePSF(FWHM, filter):
	width = 30
	sigma = FWHM / (4 * np.sqrt(2 * np.log(2)))
	PSF = np.zeros((2 * width + 1, 2 * width + 1),)
	for index1 in range(2 * width + 1):
		for index2 in range(2 * width + 1):
			PSF[index1, index2] = np.exp(-((index1 - width)**2 + (index2 - width)**2) / (2 * sigma**2))
			# 1./(2 * np.pi * sigma**2) *  (???) normalization
	empty_primary = fits.PrimaryHDU()
	image = fits.ImageHDU(PSF)
	hdulStar = fits.HDUList([empty_primary, image])
	hdulStar.writeto('image_fitting/PSF/' + filter + '.fits')
	
generatePSF(FWHM[0], filters[0])
generatePSF(FWHM[1], filters[1])
generatePSF(FWHM[2], filters[2])
generatePSF(FWHM[3], filters[3])
generatePSF(FWHM[4], filters[4])
	
