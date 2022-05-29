import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from lmfit import Model

def gaussian2D(x, A, x0, y0, sigmaX, sigmaY, angle):
	X = x[:, 0]
	Y = x[:, 1]
	
	angle = np.pi / 180. * angle
	
	a = 0.5 * (np.cos(angle) / sigmaX)**2 + 0.5 * (np.sin(angle) / sigmaY)**2
	b = 0.5 * (np.sin(2 * angle) / sigmaX**2 - np.sin(2 * angle) / sigmaY**2)
	c = 0.5 * (np.sin(angle) / sigmaX)**2 + 0.5 * (np.cos(angle) / sigmaY)**2
	
	result = A * np.exp(- a * (X - x0)**2 - b * (Y - y0) * (X - x0) - c * (Y - y0)**2)
	return result
	
def makeImage(field, imageParams, lensParams, rotation):
	A, x0, y0, sigmaX, sigmaY, angle, scale = imageParams
	X, Y = field
	dimX = len(X)
	dimY = len(Y)
	meshX, meshY = np.meshgrid(X, Y)
	x = np.zeros((dimX * dimY, 2),)
	x[:, 0] = meshX.flatten()
	x[:, 1] = meshY.flatten()
	
	galaxy = gaussian2D(x, A, x0, y0, sigmaX, sigmaY, angle)
	
	#check original image
	galaxy2D = galaxy.reshape(dimX, dimY)
	
	tension, inclination, Rs = lensParams
	inclination = inclination * np.pi / 180.
	theta = 8 * np.pi * tension * np.cos(inclination) * 206265. / scale # pixel
	lensedGalaxy = np.zeros((dimX, dimY),)
	for indexY in range(dimY):
		RsY = Rs / (1. + np.tan(inclination) * np.sin((Y[indexY] - dimY / 2) * scale / 206265.))
		Re = theta * (1 - RsY) #Einstein distance
		
		Xextended = np.linspace(-int(Re) - 1, dimX + int(Re), dimX + 2*(int(Re) + 2))
		Ypart = [Y[indexY]]
		
		meshX, meshY = np.meshgrid(Xextended, Ypart)
		cutLeft = 0
		cutRight = 0
		while(Xextended[cutLeft] < dimX / 2 + Re):
			cutLeft += 1
		while(Xextended[cutRight] < dimX / 2 - Re):
			cutRight += 1
		
		
		xLine = np.zeros((len(Xextended), 2),)
		xLine[:, 0] = meshX.flatten()
		xLine[:, 1] = meshY.flatten()
	
		galaxyLeft = gaussian2D(xLine, A, x0 - Re/2, y0, sigmaX, sigmaY, angle).reshape(len(Xextended), 1)[:, 0]
		galaxyRight = gaussian2D(xLine, A, x0 + Re/2, y0, sigmaX, sigmaY, angle).reshape(len(Xextended), 1)[:, 0]
		for indexX in range(len(Xextended)):
			if (indexX > cutLeft):
				galaxyLeft[indexX] = 0
			if (indexX < cutRight):
				galaxyRight[indexX] = 0
				
		finalRow = galaxyLeft[0:dimX] + galaxyRight[len(Xextended) - dimX - 1: len(Xextended) - 1]
		lensedGalaxy[indexY, :] = finalRow
	
	theta1 = theta * (1 - Rs / (1. - np.tan(inclination) * np.sin( dimY / 2. * scale / 206265.)))
	theta2 = theta * (1 - Rs / (1. + np.tan(inclination) * np.sin( dimY / 2. * scale / 206265.)))
	fig, (ax1, ax2) = plt.subplots(1, 2)
	
	ax1.plot([dimX/2, dimX/2], [0, dimY], color = 'white', linewidth = 1) #string
	ax1.plot([dimX/2 + theta1, dimX/2 + theta2], [0, dimY], color = 'white', linewidth = 1) 
	ax1.plot([dimX/2 - theta1, dimX/2 - theta2], [0, dimY], color = 'white', linewidth = 1)
	
	ax1.imshow(galaxy2D)
	ax1.set_title('original image')
	
	ax2.plot([dimX/2, dimX/2], [0, dimY], color = 'white', linewidth = 1) #string
	ax2.plot([dimX/2 + theta1, dimX/2 + theta2], [0, dimY], color = 'white', linewidth = 1) 
	ax2.plot([dimX/2 - theta1, dimX/2 - theta2], [0, dimY], color = 'white', linewidth = 1)
	
	ax2.imshow(lensedGalaxy)
	ax2.set_title('lensed image')
	ax2.set_xlabel(r'$G\mu = 1.37 \cdot 10^{-2}$' + '\n' + r'$i = 89.9958^\circ$')
	plt.show()
		
		
			
	

dimX = 3001
dimY = 3001
X = np.linspace(0, dimX - 1, dimX)	
Y = np.linspace(0, dimY - 1, dimY)	
scale = 0.004 #arcsec/pix
field = X, Y
A = 10.
x0 = 2100.
y0 = 1500.
sigmaX = 100.
sigmaY = 250.
angle = 10.
imageParams = A, x0, y0, sigmaX, sigmaY, angle, scale

tension = 1.7e-2 #Gmu
inclination = 89.9965 #i in degrees
Rs = 0.5 # Rs / Rd, relative distance
lensParams = tension, inclination, Rs

rotation = 0

makeImage(field, imageParams, lensParams, rotation)