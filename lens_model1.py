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
		
def makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing, realImage, doPSF):
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
	
	if (doPSF == 1):
		imageConvolved = sig.convolve2d(image, PSF, mode = 'same', boundary = 'fill')
	else:
		imageConvolved = image
	
	# pixelizated galaxy
	dimX1 = int(dimX / scale2)
	dimY1 = int(dimY / scale2)
	pixelizatedImage = np.zeros((dimX1, dimY1),)
	for indexX in range(dimX1):
		for indexY in range(dimY1):
			pixelizatedImage[indexX, indexY] = np.sum(imageConvolved[scale2 * indexX : scale2 * (indexX + 1), scale2 * indexY : scale2 * (indexY + 1)])
	
	residuals = realImage - pixelizatedImage
				
	# drawing stuff
	
	#image for playing with parameters
	if(graphing == 2):
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
				
		if (doPSF == 1):
			originalImageConvolved = sig.convolve2d(originalImage, PSF, mode = 'same', boundary = 'fill')
		else:
			originalImageConvolved = originalImage
		
		fig, ax = plt.subplots(1, 2)
		fig.set_size_inches(10,6)
		
		ax[0].imshow(originalImageConvolved)
		ax[0].contour(originalImageConvolved, levels = 3, colors = 'w')
		ax[0].set_title('original image')
		ax[0].set_xlabel(r'$\eta$, pix')
		ax[0].set_ylabel(r'$\xi$, pix')
		
		ax[1].imshow(imageConvolved)
		ax[1].contour(imageConvolved, levels = 3, colors = 'w')
		ax[1].set_title('lensed image, ' + r'$G\mu/c^2 = $' + '{0:.3f}'.format(tension) + ' i = ' + '{0:.3f}'.format(inclination * 180 / np.pi) + r'$^\circ$')
		ax[1].set_xlabel(r'$\eta$, pix')
		ax[1].set_ylabel(r'$\xi$, pix')
		
		# scale
		length = 1. / scale1 
		
		ax[0].plot([dimX / 10, dimX / 10], [dimY / 10, dimY / 10 + length], 'b', label = '1 arcsec')
		ax[1].plot([dimX / 10, dimX / 10], [dimY / 10, dimY / 10 + length], 'b', label = '1 arcsec')
		
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
		
		ax[1].plot(string[0:dimX - delCount0, 1], string[0:dimY - delCount0, 0], 'r', label = 'string')
		ax[1].plot(string1[0:dimX - delCount1, 1], string1[0:dimY - delCount1, 0], 'r--', label = 'string' + r'$\pm \theta_E/2$')
		ax[1].plot(string2[0:dimX - delCount2, 1], string2[0:dimY - delCount2, 0], 'r--')
		
		ax[0].legend()
		ax[1].legend()
		
		plt.show()
	
	# image for the fit
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
		
		if (doPSF == 1):
			originalImageConvolved = sig.convolve2d(originalImage, PSF, mode = 'same', boundary = 'fill')
		else:
			originalImageConvolved = originalImage
			
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

typeOfImage = 'gaussian'	
dimX = 200
dimY = 200
frameParams = dimX, dimY
scale1 = 0.1 #arcsec/pix 
scale2 = 1 # pix/pix
pixelizationParams = scale1, scale2
sigma = 2 * scale2 * scale1 #arcsec, PSF radius
PSF = makePSF(sigma, pixelizationParams)

A = 1
x0 = -20
y0 = 50
sigmaX = 20
sigmaY = 10
angle = 0
Rs = 0.5
rotation = 0.
shiftX = 0.
shiftY = -40.
tension = 0.01
inclination = 89.99

originalParams = A, x0, y0, sigmaX, sigmaY, angle, typeOfImage
lensParams = tension, inclination, Rs
shift = shiftX, shiftY
graphing = 2
realImage = np.zeros((dimX, dimY),)

result = makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing, realImage, doPSF = 0)


'''
# preparation of global variables to speed up the minimization
typeOfImage = 'gaussian'
filter = 'i'
FWHM = dict([('g', 9.93), ('i', 10.02), ('r', 9.38), ('y', 6.52), ('z', 7.73)])

dimX = 300
dimY = 300 
frameParams = dimX, dimY
scale1 = 0.258 / 3. #arcsec/pix 
scale2 = 3 # pix/pix
pixelizationParams = scale1, scale2
sigma = FWHM[filter] / 2.355 * scale2 * scale1 #arcsec, PSF radius
image = getRealImage(filter)
sigmaImage = getSigmaImage(filter)

def likehood(params):
	t_init = time.time()
	
	# picture parameters
	A = params[0]
	x0 = params[1]
	y0 = params[2]
	sigmaX = params[3]
	sigmaY = params[4]
	angle = params[5]
	Rs = params[6] # Rs / Rd, relative distance
	rotation = params[7]
	shiftX = params[8]
	shiftY = params[9]
	P1 = params[10]
	P2 = params[11]
	tension = np.sqrt(P1**2 + P2**2) / (8 * np.pi)
	inclination = np.arctan(P2 / P1) * 180. / np.pi

	# technical settings
	graphing = 1
	offset = 25 
	 
	originalParams = A, x0, y0, sigmaX, sigmaY, angle, typeOfImage
	lensParams = tension, inclination, Rs
	shift = shiftX, shiftY
	PSF = makePSF(sigma, pixelizationParams)
	
	lensedImage, residuals = makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing, image)
	
	chi2 = np.sum(residuals[offset - 1: dimX - offset, offset - 1: dimX - offset]**2 / sigmaImage[offset - 1: dimX - offset, offset - 1: dimY - offset]**2) / (dimX - 2 * offset) / (dimY - 2 * offset)
	
	t_final = time.time()
	print('eval. time = ' + "{:.2f}".format(t_final - t_init) + ', chi2 = ' + "{:.6f}".format(chi2) + ' (MU, I) = (' + "{:.6f}".format(tension) + ', ' + "{:.6f}".format(inclination) + ')')
	
	return chi2



#initial guess for every fit (works faster)
#0.7649154231645788 29.50613134601861 17.79305014501254 11.914492314877211 8.221272354128946 -4.801652913445048 -0.965474595002845 64.6774433517703 9.628723298279256 #-28.49988255635241 0.04923651327539106 1.5707933203848423



A = 0.764915423164578 
x0 = 29.50613134601861
y0 = 17.79305014501254
sigmaX = 11.914492314877211
sigmaY = 8.221272354128946
angle = -4.801652913445048
Rs = -0.965474595002845 # Rs / Rd, relative distance
rotation = 64.6774433517703
shiftX = 9.628723298279256
shiftY = -28.49988255635241  
tension = 0.04923651327539106 # 8e-7 Gmu is upper bound for normal theory
inclination = 1.5707933203848423 * 180. / np.pi  # i in degrees

# lensing parameters, which are suitable for gradient descent
P1 = 8 * np.pi * tension * np.cos(inclination * np.pi / 180.)
P2 = 8 * np.pi * tension * np.sin(inclination * np.pi / 180.)

init = [A, x0, y0, sigmaX, sigmaY, angle, Rs, rotation, shiftX, shiftY, P1, P2]

res = minimize(likehood, init, options={'gtol': 1e-7, 'disp': True})
print('minimization successful')

params = res.x
A = params[0] 
x0 = params[1]
y0 = params[2]
sigmaX = params[3]
sigmaY = params[4] 
angle = params[5]
Rs = params[6]
rotation = params[7] 
shiftX = params[8]
shiftY = params[9]
P1 = params[10] 
P2 = params[11]
tension = np.sqrt(P1**2 + P2**2) / (8 * np.pi)
inclination = np.arctan(P2 / P1) * 180. / np.pi

print('[A, x0, y0, sigmaX, sigmaY, angle, Rs, rotation, shiftX, shiftY, tension, inclination]')
print(A, x0, y0, sigmaX, sigmaY, angle, Rs, rotation, shiftX, shiftY, tension, inclination)

graphing = 1
	
originalParams = A, x0, y0, sigmaX, sigmaY, angle, typeOfImage
lensParams = tension, inclination, Rs 
shift = shiftX, shiftY
PSF = makePSF(sigma, pixelizationParams)

lensedImage, residuals = makeImage(frameParams, originalParams, lensParams, rotation, shift, pixelizationParams, PSF, graphing, image)

'''
