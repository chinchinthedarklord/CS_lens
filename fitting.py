import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import scipy.signal as sig
from lmfit import Model

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
		
def makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing):
	#unpacking parameters
	dimX, dimY = frameParams
	A, x0, y0, sigmaX, sigmaY, angle, typeOfImage = originalParams
	tension, inclination, Rs = lensParams
	shiftX, sgiftY = shift
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
		
			
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(10,5)
		
		ax1.imshow(originalImageConvolved)
		ax1.contour(originalImageConvolved, levels = 5, colors = 'w')
		ax1.set_title('original image')
		ax1.set_xlabel('Y')
		ax1.set_ylabel('X')
		
		ax2.imshow(imageConvolved)
		ax2.contour(imageConvolved, levels = 5, colors = 'w')
		ax2.set_title('lensed image')
		ax2.set_xlabel('Y')
		ax2.set_ylabel('X')
		
		# scale
		length = 1. / scale1 
		
		ax1.plot([dimX / 10, dimX / 10], [dimY / 10, dimY / 10 + length], 'b', label = '1 arcsec')
		ax2.plot([dimX / 10, dimX / 10], [dimY / 10, dimY / 10 + length], 'b', label = '1 arcsec')
		
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
		
		ax2.plot(string[0:dimX - delCount0, 1], string[0:dimY - delCount0, 0], 'r', label = 'string')
		ax2.plot(string1[0:dimX - delCount1, 1], string1[0:dimY - delCount1, 0], 'r--', label = 'string' + r'$\pm \theta_E/2$')
		ax2.plot(string2[0:dimX - delCount2, 1], string2[0:dimY - delCount2, 0], 'r--')
		
		ax1.legend()
		ax2.legend()
		
		plt.show()
		
		'''
		fig.savefig('gif/' + '{0:03}'.format(i) + '.png')
		plt.close(fig) 
		print(i)
		'''
		
			
	return image
	
def getRealImage(filter, X1, X2, Y1, Y2):
	image = 0
	return image

def makeResiduals(image, model):
	return 0
	
dimX = 301
dimY = 301
frameParams = dimX, dimY

A = 1.
x0 = 0.
y0 = 80.
sigmaX = 25.
sigmaY = 4.
angle = -45.
typeOfImage = 'gaussian'
originalParams = A, x0, y0, sigmaX, sigmaY, angle, typeOfImage

tension = 1e-2 #Gmu
inclination = 89.995 #i in degrees
Rs = 0.7 # Rs / Rd, relative distance
lensParams = tension, inclination, Rs

rotation = 0.
shiftX = 0.
shiftY = -50.
shift = shiftX, shiftY

scale1 = 0.02 #arcsec/pix 
scale2 = 1. #arcsec/pix
pixelizationParams = scale1, scale2

sigma = 0.3 #arcsec, PSF radius
PSF = makePSF(sigma, pixelizationParams)
graphing = 1

lensedImage = makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing)

'''
for i in range(20):
	y0 = 100 - 10 * i
	originalParams = A, x0, y0, sigmaX, sigmaY, angle, typeOfImage
	lensedImage = makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing, i)
	#plt.imshow(lensedImage, vmin = 0, vmax = 20)
	#plt.show()
'''
