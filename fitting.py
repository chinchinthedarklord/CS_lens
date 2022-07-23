import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import scipy.signal as sig
from astropy.io import fits
from scipy.optimize import minimize
from lmfit import Model
import time

def gaussian2D(x, y, A, x0, y0, sigmaX, sigmaY, angle):
	angle = np.pi / 180. * angle
	a = 0.5 * (np.cos(angle) / sigmaX)**2 + 0.5 * (np.sin(angle) / sigmaY)**2
	b = 0.5 * (np.sin(2 * angle) / sigmaX**2 - np.sin(2 * angle) / sigmaY**2)
	c = 0.5 * (np.sin(angle) / sigmaX)**2 + 0.5 * (np.cos(angle) / sigmaY)**2
	result = A * np.exp(- a * (x - x0)**2 - b * (y - y0) * (x - x0) - c * (y - y0)**2)
	return result
	
def ellipse2D(x, y, A, x0, y0, majorAxis, minorAxis, angle):
	angle = np.pi / 180. * angle
	a = (np.cos(angle) / majorAxis)**2 + (np.sin(angle) / minorAxis)**2
	b = (np.sin(2 * angle) / majorAxis**2 - np.sin(2 * angle) / minorAxis**2)
	c = (np.sin(angle) / majorAxis)**2 + (np.cos(angle) / minorAxis)**2
	if(a * (x - x0)**2 + b * (y - y0) * (x - x0) + c * (y - y0)**2 > 1):
		result = 0
	else:
		result = A
	return result
	
def obj(x, y, A, x0, y0, majorAxis, minorAxis, angle, typeOfImage):
	result = 0
	if (typeOfImage == 'gaussian'):
		result = gaussian2D(x, y, A, x0, y0, majorAxis, minorAxis, angle)
	if (typeOfImage == 'ellipse'):
		result = ellipse2D(x, y, A, x0, y0, majorAxis, minorAxis, angle)
	return result
	
def makePSF(sigma, pixelizationParams):
	scale1, scale2 = pixelizationParams
	sigma = sigma / scale1
	# PSF 5 sigma
	boundary = int(5 * sigma) * 2 + 1
	result = np.zeros((boundary, boundary),)
	for index1 in range(boundary):
		for index2 in range(boundary):
			A = 1.
			result[index1, index2] = gaussian2D(index1, index2, A, (boundary - 1) / 2, (boundary - 1) / 2, sigma, sigma, 0)
	norm = np.sum(result)
	result = result / norm
	return result
		
def makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing, realImage):
	#unpacking parameters
	dimX, dimY = frameParams
	A, x0, y0, sigmaX, sigmaY, angle, typeOfImage = originalParams
	tension, inclination, Rs = lensParams
	shiftX, shiftY = shift
	rotation = rotation * np.pi / 180.
	# scale1 - for the raw image (arcsec/pix). Lensing procedure is described in arcsec
	# scale2 - for the minimization functional (arcsec/pix).
	scale1, scale2 = pixelizationParams
	
	# coordinate grid
	X = np.linspace(-(dimX - 1) / 2, (dimX - 1) / 2, dimX)
	Y = np.linspace(-(dimY - 1) / 2, (dimY - 1) / 2, dimY)
	image = np.zeros((dimX, dimY),)
	
	inclination = inclination * np.pi / 180.
	theta = 8 * np.pi * tension * 206265. / scale1 # pixel
	
	for indexX in range(dimX):
		for indexY in range(dimY):
			# reverse shift
			x1 = X[indexX] - shiftX
			y1 = Y[indexY] - shiftY
			# reverse rotation
			x2 = x1 * np.cos(rotation) + y1 * np.sin(rotation)
			y2 = -x1 * np.sin(rotation) + y1 * np.cos(rotation)
			
			RsY = Rs / (1. + np.tan(inclination) * np.sin(x2 * scale1 / 206265.))
			Re = theta * (1 - RsY) * (np.cos(inclination) + np.sin(inclination) * x2 * scale1 / 206265. ) #Einstein distance in pixels
			
			if(np.abs(y2) <= Re):
				image[indexX, indexY] = obj(x2, y2 - Re / 2, A, x0, y0, sigmaX, sigmaY, angle, typeOfImage) + obj(x2, y2 + Re / 2, A, x0, y0, sigmaX, sigmaY, angle, typeOfImage)
			if(y2 < - Re):
				image[indexX, indexY] = obj(x2, y2 + Re / 2, A, x0, y0, sigmaX, sigmaY, angle, typeOfImage)
			if(y2 > Re):
				image[indexX, indexY] = obj(x2, y2 - Re / 2, A, x0, y0, sigmaX, sigmaY, angle, typeOfImage)
	
	imageConvolved = sig.convolve2d(image, PSF, mode = 'same', boundary = 'fill')
	
	# pixelizated galaxy
	dimX1 = int(dimX / scale2)
	dimY1 = int(dimY / scale2)
	pixelizatedImage = np.zeros((dimX1, dimY1),)
	for indexX in range(dimX1):
		for indexY in range(dimY1):
			pixelizatedImage[indexX, indexY] = np.sum(imageConvolved[scale2 * indexX : scale2 * (indexX + 1), scale2 * indexY : scale2 * (indexY + 1)])
	
	residuals = realImage - pixelizatedImage
				
	# drawing stuff
	if(graphing == 1):
		originalImage = np.zeros((dimX, dimY),)
		
		# original image
		for indexX in range(dimX):
			for indexY in range(dimY):
				# reverse shift
				x1 = X[indexX] - shiftX
				y1 = Y[indexY] - shiftY
				# reverse rotation
				x2 = x1 * np.cos(rotation) + y1 * np.sin(rotation)
				y2 = -x1 * np.sin(rotation) + y1 * np.cos(rotation)
				originalImage[indexX, indexY] = obj(x2, y2, A, x0, y0, sigmaX, sigmaY, angle, typeOfImage)
		
		originalImageConvolved = sig.convolve2d(originalImage, PSF, mode = 'same', boundary = 'fill')
		
			
		fig, ax = plt.subplots(2, 3)
		fig.set_size_inches(12,8)
		
		ax[0, 0].imshow(originalImageConvolved)
		ax[0, 0].contour(originalImageConvolved, levels = 3, colors = 'w')
		ax[0, 0].set_title('original image')
		ax[0, 0].set_xlabel('Y')
		ax[0, 0].set_ylabel('X')
		
		ax[1, 0].imshow(imageConvolved)
		ax[1, 0].contour(imageConvolved, levels = 3, colors = 'w')
		ax[1, 0].set_title('lensed image')
		ax[1, 0].set_xlabel('Y')
		ax[1, 0].set_ylabel('X')
		
		ax[0, 1].imshow(pixelizatedImage)
		ax[0, 1].contour(pixelizatedImage, levels = 3, colors = 'w')
		ax[0, 1].set_title('lensed pixelized image')
		ax[0, 1].set_xlabel('Y')
		ax[0, 1].set_ylabel('X')
		
		ax[1, 1].imshow(realImage)
		ax[1, 1].set_title('real image')
		ax[1, 1].set_xlabel('Y')
		ax[1, 1].set_ylabel('X')
		
		ax[0, 2].imshow(realImage)
		ax[0, 2].contour(pixelizatedImage, levels = 3, colors = 'w')
		ax[0, 2].set_title('real image + contours')
		ax[0, 2].set_xlabel('Y')
		ax[0, 2].set_ylabel('X')
		
		ax[1, 2].imshow(residuals)
		#ax[1, 2].contour(residuals, levels = 2, colors = 'w')
		ax[1, 2].set_title('residuals')
		ax[1, 2].set_xlabel('Y')
		ax[1, 2].set_ylabel('X')
		
		
		# scale
		length = 1. / scale1 
		
		ax[0, 0].plot([dimX / 10, dimX / 10], [dimY / 10, dimY / 10 + length], 'b', label = '1 arcsec')
		ax[1, 0].plot([dimX / 10, dimX / 10], [dimY / 10, dimY / 10 + length], 'b', label = '1 arcsec')
		
		# string position
		string = np.zeros((dimX, 2))
		string1 = np.zeros((dimX, 2))
		string2 = np.zeros((dimX, 2))
		
		delCount0 = 0
		delCount1 = 0
		delCount2 = 0
		for indexX in range(dimX):
			for indexY in range(dimY):
				# reverse shift
				x1 = X[indexX] - shiftX
				y1 = Y[indexY] - shiftY
				# reverse rotation
				x2 = x1 * np.cos(rotation) + y1 * np.sin(rotation)
				y2 = -x1 * np.sin(rotation) + y1 * np.cos(rotation)
				
				RsY = Rs / (1. + np.tan(inclination) * x2 * scale1 / 206265.)
				Re = theta * (1 - RsY) * (np.cos(inclination) + np.sin(inclination) * x2 * scale1 / 206265.)#Einstein distance in pixels
				
				if (np.abs(y2) <= 0.5):
					string[indexX - delCount0, 0] = indexX
					string[indexX - delCount0, 1] = indexY
				if (np.abs(y2 - Re) <= 0.5):
					string1[indexX - delCount1, 0] = indexX
					string1[indexX - delCount1, 1] = indexY
				if (np.abs(y2 + Re) <= 0.5):
					string2[indexX - delCount2, 0] = indexX
					string2[indexX - delCount2, 1] = indexY
			if((string[indexX - delCount0, 0] <= 0) or (string[indexX - delCount0, 0] >= dimX - 1) or (string[indexX - delCount0, 1] <= 0) or (string[indexX - delCount0, 1] >= dimY - 1)):
				delCount0 +=1
			if((string1[indexX - delCount1, 0] <= 0) or (string1[indexX - delCount1, 0] >= dimX - 1) or (string1[indexX - delCount1, 1] <= 0) or (string1[indexX - delCount1, 1] >= dimY - 1)):
				delCount1 +=1
			if((string2[indexX - delCount2, 0] <= 0) or (string2[indexX - delCount2, 0] >= dimX - 1) or (string2[indexX - delCount2, 1] <= 0) or (string2[indexX - delCount2, 1] >= dimY - 1)):
				delCount2 +=1
		
		ax[1, 0].plot(string[0:dimX - delCount0, 1], string[0:dimY - delCount0, 0], 'r', label = 'string')
		ax[1, 0].plot(string1[0:dimX - delCount1, 1], string1[0:dimY - delCount1, 0], 'r--', label = 'string' + r'$\pm \theta_E/2$')
		ax[1, 0].plot(string2[0:dimX - delCount2, 1], string2[0:dimY - delCount2, 0], 'r--')
		
		ax[0, 0].legend()
		ax[1, 0].legend()
		
		plt.show()
		
		'''
		fig.savefig('gif/' + '{0:03}'.format(i) + '.png')
		plt.close(fig) 
		print(i)
		'''	
			
	return pixelizatedImage, residuals
	
def getRealImage(filter):
	hdul = fits.open('image_fitting/galaxy/' + filter + '.fits')
	image = hdul[1].data
	hdul.close()
	return image
	
def getSigmaImage(filter):
	image = getRealImage(filter)
	dimX = len(image[:, 0])
	dimY = len(image[0, :])	
	
	# from header of original image
	# gain is approx 1
	nOfFrames = 30
	texp = 45
	
	imageCounts = nOfFrames * texp * image
	mean = np.std(imageCounts[:, 0:10])**2
	imageCounts = imageCounts + mean
	
	sigma = np.zeros((dimX, dimY),)
	for indexX in range(dimX):
		for indexY in range(dimY):
			if (imageCounts[indexX, indexY] > 0):
				sigma[indexX, indexY] = np.sqrt(imageCounts[indexX, indexY]) / (nOfFrames * texp)
			if (imageCounts[indexX, indexY] < 0):
				sigma[indexX, indexY] = np.sqrt(mean) / (nOfFrames * texp)
	return sigma

filter = 'i'
image = getRealImage(filter)
sigmaImage = getSigmaImage(filter)

def likehood(params):
	t_init = time.time()
	
	A = params[0]
	x0 = params[1]
	y0 = params[2]
	sigmaX = params[3]
	sigmaY = params[4]
	angle = params[5]
	tension = 5e-2#8e-7 #Gmu
	inclination = 89.9995 #i in degrees
	Rs = params[6] # Rs / Rd, relative distance
	rotation = params[7]
	shiftX = params[8]
	shiftY = params[9]

	#initial guess
	'''
	A = 0.5
	x0 = 30.
	y0 = 20.
	sigmaX = 12.
	sigmaY = 6.
	angle = -5.
	tension = 5e-2#8e-7 #Gmu
	inclination = 89.9995 #i in degrees
	Rs = 0.25 # Rs / Rd, relative distance
	rotation = 60.
	shiftX = 8.
	shiftY = -27.
	'''

	filter = 'i'
	graphing = 0
	
	dimX = 300
	dimY = 300
	offset = 25
	frameParams = dimX, dimY
	
	typeOfImage = 'gaussian'
	originalParams = A, x0, y0, sigmaX, sigmaY, angle, typeOfImage
	lensParams = tension, inclination, Rs

	shift = shiftX, shiftY
	transformationParams = shiftX, shiftY, rotation

	scale1 = 0.258 / 3. #arcsec/pix 
	scale2 = 3 #pix/pix
	pixelizationParams = scale1, scale2

	sigma = 2. * scale2 * scale1 #arcsec, PSF radius
	PSF = makePSF(sigma, pixelizationParams)
	
	lensedImage, residuals = makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing, image)
	
	chi2 = np.sum(residuals[offset - 1: dimX - offset, offset - 1: dimX - offset]**2 / sigmaImage[offset - 1: dimX - offset, offset - 1: dimY - offset]**2) / (dimX - 2 * offset) / (dimY - 2 * offset)
	
	t_final = time.time()
	print('eval. time = ' + "{:.2f}".format(t_final - t_init) + ', chi2 = ' + "{:.6f}".format(chi2))
	
	return chi2



#initial guess
A = 0.5
x0 = 30.
y0 = 20.
sigmaX = 12.
sigmaY = 6.
angle = -5.
tension = 5e-2#8e-7 #Gmu
inclination = 89.9995 #i in degrees
Rs = 0.25 # Rs / Rd, relative distance
rotation = 60.
shiftX = 8.
shiftY = -27.

init = [0.5, 30, 20, 12, 6, -5, 0.25, 60, 8, -27]
res = minimize(likehood, init)
print('minimization successful')
params = res.x
A = params[0]
x0 = params[1]
y0 = params[2]
sigmaX = params[3]
sigmaY = params[4]
angle = params[5]
tension = 5e-2#8e-7 #Gmu
inclination = 89.9995 #i in degrees
Rs = params[6] # Rs / Rd, relative distance
rotation = params[7]
shiftX = params[8]
shiftY = params[9]

print(params)

graphing = 1
	
dimX = 300
dimY = 300
offset = 25
frameParams = dimX, dimY
	
typeOfImage = 'gaussian'
originalParams = A, x0, y0, sigmaX, sigmaY, angle, typeOfImage
lensParams = tension, inclination, Rs

shift = shiftX, shiftY
transformationParams = shiftX, shiftY, rotation

scale1 = 0.258 / 3. #arcsec/pix 
scale2 = 3 #pix/pix
pixelizationParams = scale1, scale2

sigma = 2. * scale2 * scale1 #arcsec, PSF radius
PSF = makePSF(sigma, pixelizationParams)

lensedImage, residuals = makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing, image)


