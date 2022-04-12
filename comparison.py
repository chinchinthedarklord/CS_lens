# 07058 - положение щели на кадре
# 07059 - первая галактика
# 07060 - вторая галактика
# 07061 - FeNe, калибровка по известной лампе.
# 07034,35,62,85,86 -  BIAS

import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from lmfit import Model

import os

dim1 = 3500
dim2 = 250

def linear(x, A, B):
	return A * x + B
linear_model = Model(linear)

def quadratic(x, A, B, C):
	return A * x * x + B * x + C
quadratic_model = Model(quadratic)

# подложка
def getBIAS():
	BIAS = np.zeros((dim1, dim2))
	cnt = 0
	
	for filename in os.listdir('2022-03-07/bias'):
		file_image = fits.open('2022-03-07/bias/'+ filename)
		data_image = file_image[0].data #2D array
		data_image = data_image.astype('float64')
		file_image.close()
		BIAS = BIAS + data_image
		cnt += 1
		
	BIAS = BIAS / cnt
	
	result = BIAS
	return result

# сырой снимок	
def getRawSpectra(obj, BIAS):
	SHOT = np.zeros((dim1, dim2))
	
	file_image = fits.open('2022-03-07/'+ obj + '.fits')
	data_image = file_image[0].data #2D array
	data_image = data_image.astype('float64')
	file_image.close()
	
	SHOT = data_image - BIAS
	
	return SHOT

# удаление космических лучей	 медианным фильтром
def removeCosmics(obj, BIAS):
	SHOT = getRawSpectra(obj, BIAS)
	spectrum = np.zeros((dim1), )
	
	offset = 10
	percentile = 98
	iterations = 1
	
	for i in range(iterations):
		for index1 in range(dim1): 
			median = np.median(SHOT[index1, offset : dim2 - offset])
			quantile = np.percentile(SHOT[index1, offset : dim2 - offset], percentile)
			for index2 in range(dim2 - 2 * offset):
				if (SHOT[index1, offset + index2] > quantile):
					SHOT[index1, offset + index2] = median
			spectrum[index1] = np.mean(SHOT[index1, offset : dim2 - offset])
	
	return SHOT[:, offset : dim2 - offset]

# надо выгрузить линии из спектра лампы, усреднить по их профилям каким-нибудь полиномом, это будет условный FLAT, потом на него разделить и убирать линии атмосферы. Либо сделать фит и вычитать сигнал

#координатное представление линии, потому что она расположена криво
def getLine():
	width = 17
	coordsX = np.array([2617, 2271, 1981, 1809, 1454, 745, 484])
	coordsY = np.array([116, 118, 121, 123, 125, 128, 131])
	
	pixels = np.linspace(1, dim1, dim1)
	result = linear_model.fit(coordsY, x = coordsX, A = 0., B = 0.)
	A = result.params['A'].value
	B = result.params['B'].value
	
	return A, B, width

# срез атмосферной линии - парабола, потому что кривизна линий + плоское поле
def removeBackground(SHOT):
	offset = 80
	offsetP = 10
	
	pixels1 = np.linspace(1, offset, offset)
	pixels2 = np.linspace(dim2 - offset + 1 - offsetP * 2, dim2 - offsetP * 2, offset)
	pixels = np.concatenate((pixels1, pixels2))
	
	spectrum = np.zeros((dim1),)
	err = np.zeros((dim1),)
	lineA, lineB, width = getLine()
	
	for index1 in range(dim1):
		leftLine = SHOT[index1, 0 : offset]
		rightLine = SHOT[index1, dim2 - offsetP * 2 - offset : dim2 - offsetP * 2]
		line = np.concatenate((leftLine, rightLine))
		result = quadratic_model.fit(line, x = pixels, A = 0, B = 0, C = 0)
		A = result.params['A'].value
		B = result.params['B'].value
		C = result.params['C'].value
		pixelsFull = np.linspace(1, dim2 - offsetP * 2, dim2 - offsetP * 2)
		SHOT[index1, :] = SHOT[index1, :] - quadratic(pixelsFull, A, B, C)
		
		flatField = line - quadratic(pixels, A, B, C)
		err[index1] = np.std(flatField) * np.sqrt(width)
		
		beginLine = int(linear(index1, lineA, lineB)) - offsetP
		endLine = int(beginLine + width)
		spectrum[index1] = np.sum(SHOT[index1, beginLine : endLine])
		
		'''
		# отмечаем расположение нужной линии
		marker = 150
		SHOT[index1, beginLine] = marker
		SHOT[index1, endLine] = marker		
		'''
		
	#скользящее среднее, чтобы убирать шумы
	'''
	binning_spectrum = 10	
	pd_spectrum = pd.DataFrame(spectrum)
	rolling_spectrum = pd_spectrum.rolling(binning_spectrum, min_periods=1)
	spectrum = rolling_spectrum.mean().to_numpy()[:, 0]
	
	binning_err = 20
	pd_err = pd.DataFrame(err)
	rolling_err = pd_err.rolling(binning_err, min_periods=1)
	err = rolling_err.mean().to_numpy()[:, 0]
	'''
	
	return SHOT, spectrum, err
	
# надо попробовать использовать плоское поле и кривизну с известного снимака лампы, чтобы убрать атмосферные линии. Но это будет очень сложно

# функция 2D-профиля линии

def gaussianLine(x, A, sigma, x0, cont):
	result = cont + A / np.sqrt(2 * np.pi * sigma) * np.exp( -(x - x0)**2 / (2 * sigma**2))
	return result
gaussian_model = Model(gaussianLine)

def spectralLine(x, A, sigma, x0, cont, A1, A2, B1, B2):
	X = x[:, 0]
	Y = x[:, 1]
	
	result = (cont + A / np.sqrt(2 * np.pi * sigma) * np.exp( -(X - x0 - B1 * Y - B2 * Y**2)**2 / (2 * sigma**2))) * (1 - A1 * Y - A2 * Y**2)
	return result
line_model = Model(spectralLine)

def getFlatCurvature(SHOT, graphing):
	# подбор параметров одиночной линии
	offset = 10
	lengthX = len(SHOT[:, 0])
	lengthY = dim2 - 2 * offset
	X = np.linspace(1, lengthX, lengthX)
	Y = np.linspace(- int(dim2 / 2) + offset, int(dim2 / 2) - offset, lengthY)
	meshX, meshY = np.meshgrid(X, Y)
	
	square = np.zeros((lengthX * lengthY, 2),)
	square[:, 0] = meshX.flatten()
	square[:, 1] = meshY.flatten()
	
	central_profile = SHOT[:, int(dim2 / 2)]
	edge_profile = SHOT[:, offset]
	cutSHOT = SHOT[:, offset : dim2 - offset]
		
	# угадываем параметры гауссианы
	initialA = (np.max(SHOT)  - 150)* np.sqrt(2 * np.pi)
	initialSigma = 1.
	initialX0 = lengthX / 2
	initialCont = 150.
	central_result = gaussian_model.fit(central_profile, x = X, A = initialA, sigma = initialSigma, x0 = initialX0, cont = initialCont)
	edge_result = gaussian_model.fit(edge_profile, x = X, A = initialA, sigma = initialSigma, x0 = initialX0, cont = initialCont)
		
	if(graphing == 1):
		plt.plot(X, central_profile)
		plt.plot(X, central_result.best_fit)
		plt.show()
		
		plt.plot(X, edge_profile)
		plt.plot(X, edge_result.best_fit)
		plt.show()
			
	# угадываем параметры реальной линии
	secondaryA = central_result.params['A'].value
	secondarySigma = central_result.params['sigma'].value
	secondaryX0 = central_result.params['x0'].value
	secondaryCont = central_result.params['cont'].value
	secondaryA1 = 0.
	secondaryA2 = (secondaryA - edge_result.params['A'].value) / (dim2 / 2 - offset)**2
	secondaryB1 = 0.
	secondaryB2 = (edge_result.params['x0'].value - secondaryX0) / (dim2 / 2 - offset)**2
		
	# фитируем реальную линию
	main_result = line_model.fit(cutSHOT.transpose().flatten(), x = square, A = secondaryA, sigma = secondarySigma, x0 = secondaryX0, cont = secondaryCont, A1 = secondaryA1, A2 = secondaryA2, B1 = secondaryB1, B2 = secondaryB2)
	A = main_result.params['A'].value
	sigma = main_result.params['sigma'].value
	x0 = main_result.params['x0'].value
	cont = main_result.params['cont'].value
	A1 = main_result.params['A1'].value
	A2 = main_result.params['A2'].value
	B1 = main_result.params['B1'].value
	B2 = main_result.params['B2'].value
	
	if(graphing == 1):
		best_line = np.zeros((lengthX, dim2 - 2 * offset),)
		for index1 in range(lengthX):
			for index2 in range(dim2 - 2 * offset):
				best_line[index1, index2] = (cont + A / np.sqrt(2 * np.pi * sigma) * np.exp( -(X[index1] - x0 - B1 * Y[index2] - B2 * Y[index2]**2)**2 / (2 * sigma**2))) * (1 - A1 * Y[index2] - A2 * Y[index2]**2)
		fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
		ax1.imshow(cutSHOT, vmin = cont, vmax = np.max(SHOT))
		ax2.imshow(best_line, vmin = cont, vmax = np.max(SHOT))
		ax3.imshow(cutSHOT - best_line, vmin = - np.max(SHOT), vmax = np.max(SHOT))
		plt.show()
	return np.array([A, sigma, x0, cont, A1, A2, B1, B2])
	
def getCurvParameters(graphing):
	BIAS = getBIAS()
	FeNe = getRawSpectra('afc07061', BIAS)
	
	pixels = 3500 - np.array([866, 891, 946, 974, 1021, 1059, 1078, 1119, 1136, 1182, 1223, 1256, 1281, 1322, 1338, 1425, 1447, 1501, 1566, 1598, 1769, 1851, 1963, 2018])
	width = 10
	
	A1 = np.zeros((len(pixels)),)
	A2 = np.zeros((len(pixels)),)
	B1 = np.zeros((len(pixels)),)
	B2 = np.zeros((len(pixels)),)
	
	for i in range(len(pixels)):
		params = getFlatCurvature(FeNe[pixels[i] - width : pixels[i] + width, :], graphing = 0)
		A1[i] = params[4]
		A2[i] = params[5]
		B1[i] = params[6]
		B2[i] = params[7]
		#print(str(i+1) + ' / ' + str(len(pixels)))
		
	if(graphing == 1):
		fig, axes = plt.subplots(2, 2)
		
		axes[0, 0].plot(pixels, A1 * dim2 / 2, label = 'A1')
		axes[0, 0].grid()
		axes[0, 0].legend()
		
		axes[0, 1].plot(pixels, A2 * dim2**2 / 4, label = 'A2')
		axes[0, 1].grid()
		axes[0, 1].legend()
		
		axes[1, 0].plot(pixels, B1 * dim2 / 2, label = 'B1')
		axes[1, 0].grid()
		axes[1, 0].legend()
		
		axes[1, 1].plot(pixels, B2 * dim2**2 / 4, label = 'B2')
		axes[1, 1].grid()
		axes[1, 1].legend()
		
		plt.show()
		
	return pixels, A1, A2, B1, B2
	
def cleanLines(obj, BIAS, graphing):
	offset = 80
	offsetP = 10
	allPixelsY = np.linspace(1, dim2 -2 * offsetP , dim2 -2 * offsetP) - dim2 / 2 - offsetP
	pixels, A1, A2, B1, B2 = getCurvParameters(graphing = 0)
	
	SHOT = removeCosmics(obj, BIAS)
	spectrum = np.zeros((dim1),)
	err = np.zeros((dim1),)
	lineA, lineB, width = getLine()	
	
	# фитируем параметры кривизны поля, чтобы потом убирать с их помощью атмосферные линии
	resultA1 = quadratic_model.fit(A1, x = pixels, A = 0., B = 0., C = 0.)
	resultA2 = quadratic_model.fit(A2, x = pixels, A = 0., B = 0., C = 0.)
	resultB1 = quadratic_model.fit(B1, x = pixels, A = 0., B = 0., C = 0.)
	resultB2 = quadratic_model.fit(B2, x = pixels, A = 0., B = 0., C = 0.)
	
	if(graphing == 1):
		fig, axes = plt.subplots(2, 2)
		
		axes[0, 0].plot(pixels, A1 * dim2 / 2, label = 'A1')
		axes[0, 0].plot(pixels, resultA1.best_fit * dim2 / 2, 'k--')
		axes[0, 0].grid()
		axes[0, 0].legend()
		
		axes[0, 1].plot(pixels, A2 * dim2**2 / 4, label = 'A2')
		axes[0, 1].plot(pixels, resultA2.best_fit * dim2**2 / 4, 'k--')
		axes[0, 1].grid()
		axes[0, 1].legend()
		
		axes[1, 0].plot(pixels, B1 * dim2 / 2, label = 'B1')
		axes[1, 0].plot(pixels, resultB1.best_fit * dim2 / 2, 'k--')
		axes[1, 0].grid()
		axes[1, 0].legend()
		
		axes[1, 1].plot(pixels, B2 * dim2**2 / 4, label = 'B2')
		axes[1, 1].plot(pixels, resultB2.best_fit * dim2**2 / 4, 'k--')
		axes[1, 1].grid()
		axes[1, 1].legend()
		
		plt.show()
		
	A1_A = resultA1.params['A'].value
	A1_B = resultA1.params['B'].value
	A1_C = resultA1.params['C'].value
	
	A2_A = resultA2.params['A'].value
	A2_B = resultA2.params['B'].value
	A2_C = resultA2.params['C'].value
	
	B1_A = resultB1.params['A'].value
	B1_B = resultB1.params['B'].value
	B1_C = resultB1.params['C'].value
	
	B2_A = resultB2.params['A'].value
	B2_B = resultB2.params['B'].value
	B2_C = resultB2.params['C'].value
	
	# прогоняем это по списку атмосферных линий, удаляя лишнее маской
	
	pixels1 = np.linspace(1, offset, offset) - dim2 / 2
	pixels2 = np.linspace(dim2 - offset + 1 - offsetP * 2, dim2 - offsetP * 2, offset) - dim2 / 2
	maskPixelsY = np.concatenate((pixels1, pixels2))
	
	line_positions = np.array([57, 185, 216, 245, 255, 270, 283, 294, 305, 317, 332, 366, 375, 448, 469, 524, 555, 581, 592, 606, 617, 629, 641, 653, 666, 692, 812, 840, 870, 878, 897, 905, 920, 931, 941, 953, 978, 995, 1012, 1034, 1047, 1066, 1100, 1092, 1127, 1166, 1192, 1214, 1224, 1234, 1270, 1306, 1336, 1365, 1389, 1398, 1411, 1420, 1430, 1441, 1461, 1487, 1788, 1814, 2250, 2064, 2885]) #заполнить линии!!!
	width = 10
	sky_model = np.zeros((dim1, dim2 - 2 * offsetP),)
	
	for i in range(len(line_positions)):
		maskPixelsX = np.linspace(1, width * 2, width * 2)
		meshX, meshY = np.meshgrid(maskPixelsX, maskPixelsY)
		square = np.zeros((len(maskPixelsX) * len(maskPixelsY), 2),)
		square[:, 0] = meshX.flatten()
		square[:, 1] = meshY.flatten()
		
		cutLineLeft = SHOT[int(line_positions[i]) - width: int(line_positions[i]) + width, 0:offset]
		cutLineRight = SHOT[int(line_positions[i]) - width: int(line_positions[i]) + width, dim2 - offset - offsetP * 2 : dim2 - offsetP * 2]
		cutLine = np.concatenate((cutLineLeft, cutLineRight), axis = 1)
		
		initialA = (np.max(SHOT[int(line_positions[i]) - width: int(line_positions[i]) + width, :])  - 10)* np.sqrt(2 * np.pi)
		initialSigma = 0.5
		initialX0 = width
		initialCont = 10.
		
		central_profile = SHOT[int(line_positions[i]) - width: int(line_positions[i]) + width, int(dim2 / 2)]
		central_result = gaussian_model.fit(central_profile, x = maskPixelsX, A = initialA, sigma = initialSigma, x0 = initialX0, cont = initialCont)
		
		initialA = central_result.params['A'].value
		initialSigma = central_result.params['sigma'].value
		initialX0 = central_result.params['x0'].value
		initialCont = central_result.params['cont'].value
		
		
		initialA1 = quadratic(line_positions[i], A1_A, A1_B, A1_C)
		initialA2 = quadratic(line_positions[i], A2_A, A2_B, A2_C)
		initialB1 = quadratic(line_positions[i], B1_A, B1_B, B1_C)
		initialB2 = quadratic(line_positions[i], B2_A, B2_B, B2_C)
		'''
		params = line_model.make_params()
		params['A1'].vary = False
		params['A2'].vary = False
		params['B1'].vary = False
		params['B2'].vary = False
		'''
		line_result = line_model.fit(cutLine.transpose().flatten(), x = square, A = initialA, sigma = initialSigma, x0 = initialX0, cont = initialCont, A1 = initialA1, A2 = initialA2, B1 = initialB1, B2 = initialB2)
		#print(line_result.fit_report())
		A = line_result.params['A'].value
		sigma = line_result.params['sigma'].value
		x0 = line_result.params['x0'].value
		cont = line_result.params['cont'].value
		A1 = line_result.params['A1'].value
		A2 = line_result.params['A2'].value
		B1 = line_result.params['B1'].value
		B2 = line_result.params['B2'].value
		
		best_line = np.zeros((2 * width, dim2 - 2 * offsetP),)
		
		for index1 in range(2 * width):
			for index2 in range(dim2 - 2 * offsetP):
				best_line[index1, index2] = (cont + A / np.sqrt(2 * np.pi * sigma) * np.exp( -(maskPixelsX[index1] - x0 - B1 * allPixelsY[index2] - B2 * allPixelsY[index2]**2)**2 / (2 * sigma**2))) * (1 - A1 * allPixelsY[index2] - A2 * allPixelsY[index2]**2)
		
		if ((graphing == 1) and(line_positions[i] == 2885)):
			fig, axes = plt.subplots(3, 1)
			vmax = np.max(SHOT[int(line_positions[i]) - width: int(line_positions[i]) + width, :])
			axes[0].imshow(best_line, vmin = 0, vmax = vmax)
			axes[1].imshow(SHOT[int(line_positions[i]) - width: int(line_positions[i]) + width, :], vmin = 0, vmax = vmax)
			axes[2].imshow(SHOT[int(line_positions[i]) - width: int(line_positions[i]) + width, :] - best_line, vmin = -vmax, vmax = vmax)
			plt.show()
		print(str(i + 1) + ' / ' + str(len(line_positions)))
		SHOT[int(line_positions[i]) - width: int(line_positions[i]) + width, :] -= best_line + cont
			
	# собираем итоговый спектр из того, что осталось посередине
	SHOT, spectrum, err = removeBackground(SHOT)
	
	return SHOT, spectrum, err
	
# калибровка по спектру неона
def getCalibration(graphing):
	wavelengths = np.array([5852.49, 5881.89, 5944.83, 5975.53, 6030.00, 6074.34, 6096.16, 6143.06, 6163.59, 6217.28, 6266.49, 6304.79, 6334.43, 6382.99, 6402.25, 6508.53, 6532.88, 6598.95, 6678.28, 6717.04, 6929.47, 7032.41, 7178.85, 7245.17])
	pixels = 3500 - np.array([866, 891, 946, 974, 1021, 1059, 1078, 1119, 1136, 1182, 1223, 1256, 1281, 1322, 1338, 1425, 1447, 1501, 1566, 1598, 1769, 1851, 1963, 2018])
	result = linear_model.fit(wavelengths, x = pixels, A = 0., B = 0.)
	A = result.params['A'].value
	B = result.params['B'].value
	
	if (graphing == 1):	
		plt.plot(pixels, wavelengths, 'ro')
		plt.plot(pixels, A* pixels + B)
		plt.xlabel('pixel number')
		plt.ylabel('wavelength, A')
		plt.grid()
		plt.show()

	z = 0.268
	lambdas = (np.linspace(1, dim1, dim1) * A + B)
	
	return lambdas, A, B

BIAS = getBIAS()
lambdas, A, B = getCalibration(graphing = 1)

#--------------------------------------
'''
SHOT, spectrum, err = cleanLines('afc07059', BIAS, graphing = 0)
plt.imshow(SHOT, vmin = -10, vmax = 30)
plt.show()
'''
#--------------------------------------

galaxy1 = removeCosmics('afc07059', BIAS)
cleanSHOT1, galaxySpectrum1, err1 = cleanLines('afc07059', BIAS, graphing = 1)

galaxy2 = removeCosmics('afc07060', BIAS)
cleanSHOT2, galaxySpectrum2, err2 = cleanLines('afc07060', BIAS, graphing = 0)

# показать очищенные 2D-спектры
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cleanSHOT1, vmin = -10, vmax = 30)
ax2.imshow(cleanSHOT2, vmin = -10, vmax = 30)
plt.show()

#показать готовые 1D-спектры
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.fill_between(lambdas, y1 = galaxySpectrum1 - err1, y2 = galaxySpectrum1 + err1, alpha = 0.4)
ax1.plot(lambdas, galaxySpectrum1, label = 'afc07059')
ax2.fill_between(lambdas, y1 = galaxySpectrum2 - err2, y2 = galaxySpectrum2 + err2, alpha = 0.4)
ax2.plot(lambdas, galaxySpectrum2, label = 'afc07060')

ax3.plot(lambdas, galaxySpectrum2 - galaxySpectrum1)

ax1.grid()
ax2.grid()
ax3.grid()

ax1.legend()
ax2.legend()

ax1.set_title('1D spectra')
ax1.set_ylabel('Intensity')
ax2.set_ylabel('Intensity')
ax3.set_xlabel(r'$\lambda, A$')

plt.show()

file_spectrum1 = open('afc07059spectrum.dat', 'w')
for i in range(len(lambdas)):
	file_spectrum1.write(str(lambdas[i]) + '\t' + str(galaxySpectrum1[i]) + '\t' + str(err1[i]) + '\n')
file_spectrum1.close()

file_spectrum2 = open('afc07060spectrum.dat', 'w')
for i in range(len(lambdas)):
	file_spectrum2.write(str(lambdas[i]) + '\t' + str(galaxySpectrum2[i]) + '\t' + str(err2[i]) + '\n')
file_spectrum2.close()

