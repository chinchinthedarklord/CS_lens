import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from scipy.integrate import odeint
from mpl_toolkits import mplot3d
from lmfit import Model
import ray

ray.init()

def linear(x, A, B):
	return A * x + B
	
linear_model = Model(linear)

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
	
#solution starts here

def HupDa(x, y, a, stringParams):
	deficit, r, theta = stringParams
	result = deficit  * (a - 1 + r) / ((x * np.cos(theta) - y * np.sin(theta))**2 + (a - 1 + r)**2) * (1 + (x * np.sin(theta) + y * np.cos(theta)) / np.sqrt(x**2 + y**2 + (a - 1 + r)**2)) / (2 * np.pi)
	return result
	
def HdownDa(x, y, a, stringParams):
	deficit, r, theta = stringParams
	result = deficit  * (a - 1 + r) / (x**2 + (a - 1 + r)**2) * (1 - y / np.sqrt(x**2 + y**2 + (a - 1 + r)**2)) / (2 * np.pi)
	return result

def HdownDy(x, y, a, stringParams):
	deficit, r, theta = stringParams
	result = deficit  * 1 / np.sqrt(x**2 + y**2 + (a - 1 + r)**2) / (2 * np.pi)
	return result
	
def HdownDx(x, y, a, stringParams):
	deficit, r, theta = stringParams
	result = deficit  * x / ((a - 1 + r)**2 + x**2) * (1 - y / np.sqrt(x**2 + y**2 + (a - 1 + r)**2)) / (2 * np.pi)
	return result
	
def HupDx(x, y, a, stringParams):
	deficit, r, theta = stringParams
	result = -np.sin(theta) * HdownDy(x, y, a, stringParams) + deficit  * np.cos(theta) * (x * np.cos(theta) - y * np.sin(theta)) / ((a - 1 + r)**2 + (x * np.cos(theta) - y * np.sin(theta))**2) * (1 + (x * np.sin(theta) + y * np.cos(theta)) / np.sqrt(x**2 + y**2 + (a - 1 + r)**2)) / (2 * np.pi)
	return result
	
def HupDy(x, y, a, stringParams):
	deficit, r, theta = stringParams
	result = -np.cos(theta) * HdownDy(x, y, a, stringParams) - deficit  * np.sin(theta) * (x * np.cos(theta) - y * np.sin(theta)) / ((a - 1 + r)**2 + (x * np.cos(theta) - y * np.sin(theta))**2) * (1 + (x * np.sin(theta) + y * np.cos(theta)) / np.sqrt(x**2 + y**2 + (a - 1 + r)**2)) / (2 * np.pi)
	return result
	
def offsetVector(x, y, a, stringParams):
	AvX = 0.5 * (HdownDx(x, y, a, stringParams) + HupDx(x, y, a, stringParams))
	AvY = 0.5 * (HdownDy(x, y, a, stringParams) + HupDy(x, y, a, stringParams))
	AnX = 0
	AnY = 0
	A = np.array([AvX, AvY, AnX, AnY])
	return A
	
def linearResponseMatrix(x, y, a, stringParams):
	deficit, r, theta = stringParams
	BvXX = np.cos(theta)**2 * HupDa(x, y, a, stringParams)+ HdownDa(x, y, a, stringParams)
	BvXY = np.sin(theta) * np.cos(theta) * HupDa(x, y, a, stringParams)
	BvYY = np.sin(theta)**2 * HupDa(x, y, a, stringParams) 
	B = np.array([[BvXX, BvXY, 0, 0],
				  [BvXY, BvYY, 0, 0],
				  [-1, 0, 0, 0],
				  [0, -1, 0, 0]])
	return B
	
def RHS(v, t, deficit, r, theta):
	stringParams = deficit, r, theta
	A = offsetVector(v[2], v[3], t, stringParams)
	B = linearResponseMatrix(v[2], v[3], t, stringParams)
	result = - A - np.matmul(B, v)
	return result
	
def singleShotSolution(initialPhaseVector, stringParams, graphing):
	deficit, r, theta = stringParams
	offset = deficit * 2
	N1 = 51
	N2 = 101
	t_before = np.linspace(1, 1 - r + offset, N1)
	t_real = np.linspace(1 - r + offset, 1 - r - offset, N2)
	t_after = np.linspace(1 - r - offset, 0, N1)
	sol1 = odeint(RHS, initialPhaseVector, t_before, args = (deficit, r, theta), hmin = 1e-11)
	sol2 = odeint(RHS, sol1[-1, :], t_real, args = (deficit, r, theta), hmin = 1e-11)
	sol3 = odeint(RHS, sol2[-1, :], t_after, args = (deficit, r, theta), hmin = 1e-11)
	result = -np.array([sol3[-1, 2], sol3[-1, 3]])
	
	#sol1 = RK45(RHS, 1.0, initialPhaseVector, 0.0, maxstep = 0.01, rtol = 0.001, vectorized = True)
	
	
	if (graphing == 1):
		fig, ax = plt.subplots(1, 2)
		fig.set_size_inches(12,6)
		
		ax[0].plot(t_before, sol1[:, 0], 'r', label = r'$v_x$')
		ax[0].plot(t_real, sol2[:, 0], 'r')
		ax[0].plot(t_after, sol3[:, 0], 'r')
		
		ax[0].plot(t_before, sol1[:, 1], 'b', label = r'$v_y$')
		ax[0].plot(t_real, sol2[:, 1], 'b')
		ax[0].plot(t_after, sol3[:, 1], 'b')
		
		ax[0].set_title('direction dependence')
		ax[0].set_xlabel(r'$a = z/R_g$')
		ax[0].set_ylabel(r'$v_i$')
		ax[0].grid()
		ax[0].legend()
		
		ax[1].plot(sol1[:, 2], sol1[:, 3], 'r', label = r'$(n_x, n_y)$')
		ax[1].plot(sol2[:, 2], sol2[:, 3], 'r')
		ax[1].plot(sol3[:, 2], sol3[:, 3], 'r')
		ax[1].set_title('path in picture plane')
		ax[1].set_xlabel(r'$n_x = x/R_g$')
		ax[1].set_ylabel(r'$n_y = y/R_g$')
		ax[1].grid()
		ax[1].legend()
		
		plt.show()
	return result

def metric(initialPositionVector, calculatedPositionVector):
	return np.sum((initialPositionVector - calculatedPositionVector)**2)
	
def shootingMethod(initialPositionVector, centerOfSearch, offset, stringParams, graphMetric, doubleSearch):
	deficit, r, theta = stringParams
	vX0 = centerOfSearch[0]
	vY0 = centerOfSearch[1]
	N = 30
	doubling = 1.
	
	vXarray = np.linspace(vX0 - offset, vX0 + offset, N)
	vYarray = np.linspace(vY0 - offset, vY0 + offset, N)
	metricArray = np.zeros((N, N))
	minimal_val1 = 1.
	minimal_val2 = 1.
	
	for indexX in range(N):
		for indexY in range(N):
			initialPhaseVector = np.array([vXarray[indexX], vYarray[indexY], 0., 0.])
			graphing = 0
			calculatedPositionVector = singleShotSolution(initialPhaseVector, stringParams, graphing)
			metricArray[indexX, indexY] = metric(initialPositionVector, calculatedPositionVector) / deficit**2
	
	ind = np.unravel_index(metricArray.argmin(), metricArray.shape)
	print('first image metric = '+ str(metricArray[ind]))
	#print(ind)
	result = np.zeros((2, 2))
	result[0, :] = np.array([vXarray[ind[0]], vYarray[ind[1]]])
	
	if (doubleSearch == 1):
		'''
		minimal_val = metricArray[ind]
		mask = 2
		metricArrayDouble = np.zeros((N, N))
		for indexX in range(N):
			for indexY in range(N):
				metricArrayDouble[indexX, indexY] = metricArray[indexX, indexY]
		
		maxX = np.min([N - 1, ind[0] + mask + 1])	
		minX = np.max([0, ind[0] - mask])
		maxY = np.min([N - 1, ind[1] + mask + 1]) 
		minY = np.max([0, ind[1] - mask])
		metricArrayDouble[minX:maxX, minY:maxY] = 10 * np.ones((maxX - minX, maxY - minY))
		'''
		isFirstMinimumRight = 0
		if ((result[0, 0] > vX0) and (result[0, 0] * np.cos(theta) - result[0, 1] * np.sin(theta) > vX0 * np.cos(theta) - vY0 * np.sin(theta))):
			isFirstMinimumRight = 1
		
		metricArrayDouble = np.zeros((N, N))
		for indexX in range(N):
			for indexY in range(N):
				metricArrayDouble[indexX, indexY] = metricArray[indexX, indexY]
				if (isFirstMinimumRight == 1):
					if ((vXarray[indexX] > vX0) and (vXarray[indexX] * np.cos(theta) - vYarray[indexY] * np.sin(theta) > vX0 * np.cos(theta) - vY0 * np.sin(theta))):
						metricArrayDouble[indexX, indexY] = 10.
				else:
					if ((vXarray[indexX] <= vX0) or (vXarray[indexX] * np.cos(theta) - vYarray[indexY] * np.sin(theta) <= vX0 * np.cos(theta) - vY0 * np.sin(theta))):
						metricArrayDouble[indexX, indexY] = 10.
						
		minimal_val1 = metricArray[ind]
		if (metricArrayDouble.min() < 1): #5 * minimal_val1):
			doubling = 2
			ind = np.unravel_index(metricArrayDouble.argmin(), metricArrayDouble.shape)
			result[1, :] = np.array([vXarray[ind[0]], vYarray[ind[1]]])
		minimal_val2 = metricArrayDouble.min()
		print('second image metric = '+ str(minimal_val2))
			
	if (graphMetric == 1):
		vYarrayMesh, vXarrayMesh = np.meshgrid(vYarray, vXarray)
		fig, ax = plt.subplots(2)
		z_min, z_max = 0, np.max(metricArray)
		c = ax[0].pcolormesh(vXarrayMesh / deficit, vYarrayMesh / deficit, metricArrayDouble, cmap='jet', vmin=z_min, vmax=z_max, shading='auto')
		ax[0].plot(np.array([vX0, result[0, 0]]) / deficit, np.array([vY0, result[0, 1]]) / deficit, 'k--')
		if (doubling == 2):
			ax[0].plot(np.array([vX0, result[1, 0]]) / deficit, np.array([vY0, result[1, 1]]) / deficit, 'k--')
		ax[0].plot([vX0 / deficit], [vY0 / deficit], 'ko', markersize = 5)
		ax[0].set_title('metric heatmap')
		# set the limits of the plot to the limits of the data
		ax[0].axis([vXarrayMesh.min() / deficit, vXarrayMesh.max() / deficit, vYarrayMesh.min() / deficit, vYarrayMesh.max() / deficit])
		ax[0].set_xlabel('X, def')
		ax[0].set_ylabel('Y, def')
		fig.colorbar(c, ax=ax)
		ax[1].plot(vXarrayMesh / deficit, metricArray[:, int(N/2)], 'r')
		ax[1].plot(vXarrayMesh / deficit, np.zeros(N), 'k--')
		plt.show()
		
	metrics = [minimal_val1, minimal_val2]
	
	return result, doubling, metrics
	
@ray.remote	
def recursiveShootingMethod(initialPositionVector, stringParams, nIter, graphMetric):
	deficit, r, theta = stringParams
	centerOfSearch = initialPositionVector
	offset = deficit * 1.5
	scaling = 5.
	iters = np.arange(nIter + 1) * np.log(scaling)
	
	
	vals, doubling, metrics = shootingMethod(initialPositionVector, centerOfSearch, offset, stringParams, graphMetric = 0, doubleSearch = 1)
	offset = offset / scaling
	
	accuracy1 = np.zeros(nIter + 1)
	accuracy2 = np.zeros(nIter + 1)
	accuracy1[0] = metrics[0]
	accuracy2[0] = metrics[1]
	
	if (doubling == 1):
		center1 = vals[0, :]
		center2 = np.zeros((2))
		for i in range(nIter):
			vals1, doubling1, metrics1 = shootingMethod(initialPositionVector, center1, offset, stringParams, graphMetric = 0, doubleSearch = 0)
			center1 = vals1[0, :]
			accuracy1[i + 1] = metrics1[0]
			offset = offset / scaling
			print(i)
		center1 = vals1[0, :]
		result1 = linear_model.fit(np.log(accuracy1), x = iters, A = 0., B = 0.)
		slope1 = result1.params['A'].value
		dslope1 = result1.params['A'].stderr
		print('single image accuracy slope')
		print(slope1, dslope1)
		
		if (graphMetric == 1):
			x0 = centerOfSearch[0]
			y0 = centerOfSearch[1]
			radiusX = 2.
			radiusY = 2.
			x1 = 0
			x2 = 0
			y1 = -radiusY
			y2 = 0
			if (theta < np.arctan(radiusX / radiusY)):
				x3 = radiusY * np.tan(theta)
				y3 = radiusY
			else:
				x3 = radiusX
				y3 = radiusX / np.tan(theta)
			plt.plot([radiusX, radiusX, -radiusX, -radiusX], [-radiusY, radiusY, -radiusY, radiusY], 'wo')
			plt.plot([x1, x2, x3], [y1, y2, y3], 'g--')
			plt.plot([x0 / deficit, vals[0, 0] / deficit], [y0 / deficit, vals[0, 1] / deficit], 'k')
			plt.plot([x0 / deficit], [y0 / deficit], 'ro', markersize = 5)
			plt.plot([vals[0, 0] / deficit], [vals[0, 1] / deficit], 'ko', markersize = 5)
			plt.grid()
			plt.show()		
			
	if(doubling == 2):
		center1 = vals[0, :]
		center2 = vals[1, :]
		
		for i in range(nIter):
			vals1, doubling1, metrics1 = shootingMethod(initialPositionVector, center1, offset, stringParams, graphMetric = 0, doubleSearch = 0)
			vals2, doubling2, metrics2 = shootingMethod(initialPositionVector, center2, offset, stringParams, graphMetric = 0, doubleSearch = 0)
			accuracy1[i + 1] = metrics1[0]
			accuracy2[i + 1] = metrics2[0]
			center1 = vals1[0, :]
			center2 = vals2[0, :]
			offset = offset / scaling
			print(i)
		
		result1 = linear_model.fit(np.log(accuracy1), x = iters, A = 0., B = 0.)
		result2 = linear_model.fit(np.log(accuracy2), x = iters, A = 0., B = 0.)
		slope1 = result1.params['A'].value
		dslope1 = result1.params['A'].stderr
		slope2 = result2.params['A'].value
		dslope2 = result2.params['A'].stderr
		print('double image accuracy slope')
		print(slope1, dslope1)
		print(slope2, dslope2)
		
		if (slope2 <= 0.3):
			print('second image convergence aborted')
			doubling = 1
		
		if (graphMetric == 1):
			x0 = centerOfSearch[0]
			y0 = centerOfSearch[1]
			radiusX = 2.		
			radiusY = 2.
			x1 = 0
			x2 = 0
			y1 = -radiusY
			y2 = 0
			if (theta < np.arctan(radiusX / radiusY)):
				x3 = radiusY * np.tan(theta)
				y3 = radiusY
			else:
				x3 = radiusX
				y3 = radiusX / np.tan(theta)
			plt.plot([radiusX, radiusX, -radiusX, -radiusX], [-radiusY, radiusY, -radiusY, radiusY], 'wo')
			plt.plot([x1, x2, x3], [y1, y2, y3], 'g--')
			plt.plot([x0 / deficit, vals1[0, 0] / deficit], [y0 / deficit, vals1[0, 1] / deficit], 'k')
			plt.plot([x0 / deficit, vals2[0, 0] / deficit], [y0 / deficit, vals2[0, 1] / deficit], 'k')
			plt.plot([x0 / deficit], [y0 / deficit], 'ro', markersize = 5)
			plt.plot([vals1[0, 0] / deficit], [vals1[0, 1] / deficit], 'ko', markersize = 5)
			plt.plot([vals2[0, 0] / deficit], [vals2[0, 1] / deficit], 'ko', markersize = 5)
			plt.grid()
			plt.show()
	new_coords1 = center1
	new_coords2 = center2
	result = np.array([doubling, new_coords1[0], new_coords1[1], new_coords2[0], new_coords2[1]])
	return result
	
#def doLensing(stringParams, frameX, frameY, Nx, Ny):
	#initialImage = np.zeros(Nx, Ny)
	#for indexX in range(Nx):
	#	for indexY in range(Ny):
	#initialImage = ellipse2D(x, y, A, x0, y0, majorAxis, minorAxis, angle)
	
def testingLensing(stringParams):
	deficit, r, theta = stringParams
	N = 16
	x = np.linspace(-0.1, 0.1, N) * deficit
	y = np.linspace(-1., 1., N) * deficit
	output_ids = []
	nIter = 5
	graphMetric = 0
	for i in range(N):
		initialPositionVector = np.array([x[i], y[i]])
		output_ids.append(recursiveShootingMethod.remote(initialPositionVector, stringParams, nIter, graphMetric))
		print(i)
	# Get results when ready.
	output_list = ray.get(output_ids)
	
	k = 206265.
	frameX = 5
	frameY = 5
	x1 = 0
	x2 = 0
	y1 = -frameY
	y2 = 0
	if (theta < np.arctan(frameX / frameY)):
		x3 = frameY * np.tan(theta)
		y3 = frameY
	else:
		x3 = frameX
		y3 = frameX / np.tan(theta)
		
	plt.plot([frameX, frameX, -frameX, -frameX], [frameY, -frameY, frameY, -frameY], 'wo', markersize = 1)
	plt.plot([x1, x2, x3], [y1, y2, y3], 'g--', label = 'string')
	for i in range(N):
		result = output_list[i]
		doubling = result[0]
		image1 = np.array([result[1], result[2]])
		image2 = np.array([result[3], result[4]])
		if (i == 0):
			plt.plot([x[i] * k], [y[i] * k], 'bo', label = 'initial point sources')
		else:
			plt.plot([x[i] * k], [y[i] * k], 'bo')
		plt.plot([x[i] * k, image1[0] * k], [y[i] * k, image1[1] * k], 'k')
		if (i == 0):
			plt.plot([image1[0] * k], [image1[1] * k], 'ro', label = 'lensed point sources')
		else:
			plt.plot([image1[0] * k], [image1[1] * k], 'ro')
		if (doubling == 2):
			plt.plot([x[i] * k, image2[0] * k], [y[i] * k, image2[1] * k], 'k')
			plt.plot([image2[0] * k], [image2[1] * k], 'ro')
	plt.xlabel(r'$\eta$' + ', arcsec')
	plt.ylabel(r'$\xi$' + ', arcsec')
	plt.grid()
	plt.legend()
	plt.show()
	return 0

r = 0.5
tension = 7e-7
deficit = 8 * np.pi * tension
theta = 10. * np.pi / 180.		
stringParams = deficit, r, theta

#initialPhaseVector = np.array([0.5 * deficit, 1. * deficit, 0., 0.])
#graphing = 1
#print(singleShotSolution(initialPhaseVector, stringParams, graphing))

# пробный метод стрельбы для одной точки
# --------------------------------------
#initialPositionVector = np.array([0.1 * deficit, -1.0 * deficit])
#graphMetric = 1
#nIter = 5
#print(recursiveShootingMethod(initialPositionVector, stringParams, nIter, graphMetric))

r = 0.5
tension = 7e-7
deficit = 8 * np.pi * tension
theta = 10. * np.pi / 180.		
stringParams = deficit, r, theta
print(testingLensing(stringParams))

r = 0.5
tension = 7e-7
deficit = 8 * np.pi * tension
theta = 0. * np.pi / 180.		
stringParams = deficit, r, theta
print(testingLensing(stringParams))

r = 0.5
tension = 7e-7
deficit = 8 * np.pi * tension
theta = 40. * np.pi / 180.		
stringParams = deficit, r, theta
print(testingLensing(stringParams))




	

	