import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from lmfit import Model

object1 = 'afc07059'
object2 = 'afc07060'

dim = 3500

def calculateCorrelations(spectrum1, spectrum2):
	rPearson, pPearson = stats.pearsonr(spectrum1, spectrum2)
	rSpearman, pSpearman = stats.spearmanr(spectrum1, spectrum2)
	rKendall, pKendall = stats.kendalltau(spectrum1, spectrum2)
	corrs = np.array([rPearson, rSpearman, rKendall])
	pvalues = np.array([pPearson, pSpearman, pKendall])
	
	return corrs, pvalues
	
def rollingMean(spectrum, N):
	pd_spectrum = pd.DataFrame(spectrum)
	rolling_spectrum = pd_spectrum.rolling(N, min_periods=1)
	return rolling_spectrum.mean().to_numpy()[:, 0]
	
def gaussian(x, cont, A, sigma, x0):
	result = cont + A * np.exp(-(x - x0)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
	return result
gaussian_model = Model(gaussian)


file1 = open(object1 + 'spectrum.dat')
pixels = np.linspace(0, dim - 1, dim)
lambdas = np.zeros((dim),)
spectrum1 = np.zeros((dim),)
err1 = np.zeros((dim),)
cnt = 0
for line in file1:
	vals = np.array([float(value) for value in line.strip().split('\t')])
	lambdas[cnt] = vals[0]
	spectrum1[cnt] = vals[1]
	err1[cnt] = vals[2]
	cnt += 1
file1.close()

file2 = open(object2 + 'spectrum.dat')
spectrum2 = np.zeros((dim),)
err2 = np.zeros((dim),)
cnt = 0
for line in file2:
	vals = np.array([float(value) for value in line.strip().split('\t')])
	spectrum2[cnt] = vals[1]
	err2[cnt] = vals[2]
	cnt += 1
file2.close()

# убираю последнюю сильную линию
spectrum2[2898] = 0.
spectrum2[2899] = 0.

# вычисление корреляций

spectrumMean1 = rollingMean(spectrum1, 3)
spectrumMean2 = rollingMean(spectrum2, 3)
errMean1 = rollingMean(err1, 3)
errMean2 = rollingMean(err2, 3)
print(calculateCorrelations(spectrumMean1, spectrumMean2))

fig, (ax1, ax2) = plt.subplots(2, 1)
pixels = np.linspace(0, dim - 1, dim)
ax1.plot(pixels, spectrumMean1)
ax2.plot(pixels, spectrumMean2)
plt.show()

# отношение 2 спектров

ratio = spectrumMean1 / spectrumMean2
ratioSigma = rollingMean(np.abs(ratio) * np.sqrt((errMean1 / spectrumMean1)**2 + (errMean2 / spectrumMean2)**2), 3)

plt.plot(lambdas, ratio, markersize = 1)
plt.fill_between(lambdas, y1 = ratio + 3 * ratioSigma, y2 = ratio - 3 * ratioSigma, alpha = 0.5)
plt.ylabel('ratio')
plt.xlabel(r'$\lambda, A$')
plt.grid()
plt.show()

# распределение отношения 2 спектров

ratioSorted = np.sort(ratio)
indexMin = 0
indexMax = 0
for i in range(len(ratio)):
	if(ratioSorted[i] > -2):
		indexMin = i
		break
		
for i in range(len(ratio)):
	if(ratioSorted[i] > 4):
		indexMax = i
		break
		
mu, std = stats.norm.fit(ratioSorted[indexMin : indexMax])
ratioX = np.linspace(-2, 4, 100)
bestFit = stats.norm.pdf(ratioX, mu, std)
plt.hist(ratioSorted[indexMin : indexMax], bins = np.linspace(-2, 4, 30), density = True, alpha = 0.6, label = 'histogram')
plt.plot(ratioX, bestFit, 'k', label = 'best fit')
plt.xlabel('ratio between spectra')
plt.ylabel('PDF')
plt.legend()
plt.grid()
title = 'best fit results: mu = %.2f' % (mu)
plt.title(title)
plt.show()

# водородные линии

alphaBegin = 734
alphaEnd = 765

betaBegin = 2318
betaEnd = 2335

# металлические линии

OIIIBegin = 2209
OIIIEnd = 2247

OIBegin = 1015
OIEnd = 1035

NIIBegin = 705
NIIEnd = 727

SII1Begin = 602
SII1End = 613

SII2Begin = 575
SII2End = 593

def fitGalaxies(lambdas, spectrum1, spectrum2, lineBegin,lineEnd, lineName):
	lambdaInterval = lambdas[lineBegin: lineEnd]
	spectrum1Interval = spectrum1[lineBegin: lineEnd]
	spectrum2Interval = spectrum2[lineBegin: lineEnd]
	
	initialCont = 10
	initialSigma = 1
	
	result_galaxy1 = gaussian_model.fit(spectrum1Interval, x = lambdaInterval, cont = initialCont, A = (np.max(spectrum1Interval) - initialCont * 3) / 2, sigma = initialSigma, x0 = lambdas[int(lineBegin / 2 + lineEnd / 2)])
	galaxy1Cont = result_galaxy1.params['cont'].value
	galaxy1A = result_galaxy1.params['A'].value
	galaxy1Sigma = result_galaxy1.params['sigma'].value
	galaxy1X0= result_galaxy1.params['x0'].value
	
	result_galaxy2 = gaussian_model.fit(spectrum2Interval, x = lambdaInterval, cont = initialCont, A = (np.max(spectrum2Interval) - initialCont), sigma = initialSigma, x0 = lambdas[int(lineBegin / 2 + lineEnd / 2)])
	galaxy2Cont = result_galaxy2.params['cont'].value
	galaxy2A = result_galaxy2.params['A'].value
	galaxy2Sigma = result_galaxy2.params['sigma'].value
	galaxy2X0= result_galaxy2.params['x0'].value
	
	lambdaBest = np.linspace(lambdas[lineBegin], lambdas[lineEnd], 100)
	fig, axes = plt.subplots(1, 2)
	
	axes[0].plot(lambdaInterval, spectrum1Interval, label = lineName + ' galaxy 1')
	axes[0].plot(lambdaBest, gaussian(lambdaBest, galaxy1Cont, galaxy1A, galaxy1Sigma, galaxy1X0), 'k--', label = 'best fit')
	axes[0].set_xlabel(r'$\lambda, A$')
	axes[0].set_ylabel('Intensity')
	axes[0].grid()
	axes[0].legend()

	axes[1].plot(lambdaInterval, spectrum2Interval, label = lineName + ' galaxy 2')
	axes[1].plot(lambdaBest, gaussian(lambdaBest, galaxy2Cont, galaxy2A, galaxy2Sigma, galaxy2X0), 'k--', label = 'best fit')
	axes[1].set_xlabel(r'$\lambda, A$')
	axes[1].set_ylabel('Intensity')
	axes[1].grid()
	axes[1].legend()
	
	plt.show()
	
	return galaxy1A, galaxy2A

alpha1, alpha2 = fitGalaxies(lambdas, spectrumMean1, spectrumMean2, alphaBegin, alphaEnd, r'$H_{\alpha}$')
beta1, beta2 = fitGalaxies(lambdas, spectrumMean1, spectrumMean2, betaBegin, betaEnd, r'$H_{\beta}$')
OIII1, OIII2 = fitGalaxies(lambdas, spectrumMean1, spectrumMean2, OIIIBegin, OIIIEnd, '[OIII]')
OI1, OI2 = fitGalaxies(lambdas, spectrumMean1, spectrumMean2, OIBegin, OIEnd, '[OI]')
NII1, NII2 = fitGalaxies(lambdas, spectrumMean1, spectrumMean2, NIIBegin, NIIEnd, '[NII]')
SII11, SII12 = fitGalaxies(lambdas, spectrumMean1, spectrumMean2, SII1Begin, SII1End, '[SII]' + r'$\lambda 6718 A$')
SII21, SII22 = fitGalaxies(lambdas, spectrumMean1, spectrumMean2, SII2Begin, SII2End, '[SII]' + r'$\lambda 6733 A$')

print(np.log(OIII1 / beta1) / np.log(10), np.log(NII1 / alpha1)/ np.log(10))
print(np.log(OIII2 / beta2) / np.log(10), np.log(NII2 / alpha2) / np.log(10))

print(np.log(OIII1 / beta1) / np.log(10), np.log((SII11 + SII21) / alpha1) / np.log(10))
print(np.log(OIII2 / beta2) / np.log(10), np.log((SII12 + SII22) / alpha2) / np.log(10))

print(np.log(OIII1 / beta1) / np.log(10), np.log(OI1 / alpha1) / np.log(10))
print(np.log(OIII2 / beta2) / np.log(10), np.log(OI2 / alpha2) / np.log(10))

# хи-квадрат для 4 линий - Halpha, Hbeta, OIII x2
alphaBegin = 730
alphaEnd = 770

betaBegin = 2200
betaEnd = 2350

Halpha1 = spectrumMean1[alphaBegin:alphaEnd]
Hbeta1 = spectrumMean1[betaBegin:betaEnd]
errHalpha1 = errMean1[alphaBegin:alphaEnd]
errHbeta1 = errMean1[betaBegin:betaEnd]

Halpha2 = spectrumMean2[alphaBegin:alphaEnd]
Hbeta2 = spectrumMean2[betaBegin:betaEnd]
errHalpha2 = errMean2[alphaBegin:alphaEnd]
errHbeta2 = errMean2[betaBegin:betaEnd]

cutSpectrum1 = np.concatenate((Halpha1, Hbeta1))
cutSpectrum2 = np.concatenate((Halpha2, Hbeta2))
cutErr1 = np.concatenate((errHalpha1, errHbeta1))
cutErr2 = np.concatenate((errHalpha2, errHbeta2))
DOF = len(cutSpectrum1)

# вычисление хи2
chi2_1 = np.sum((cutSpectrum2 - cutSpectrum1)**2 / (cutErr1**2))
print(chi2_1, chi2_1 / DOF, DOF)

# BPT диаграмма










