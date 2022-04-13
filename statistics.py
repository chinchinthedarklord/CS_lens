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
	result = cont + A * np.exp(-(x - x0)**2 / (2 * sigma**2))
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

# Индивидуальные линии

alphaBegin = 734
alphaEnd = 765

betaBegin = 2318
betaEnd = 2335

# OIII линии
'''
alphaBegin = 2209
alphaEnd = 2247

betaBegin = 2258
betaEnd = 2282
'''

HalphaLambda = lambdas[alphaBegin:alphaEnd]
HbetaLambda = lambdas[betaBegin:betaEnd]

Halpha1 = spectrumMean1[alphaBegin:alphaEnd]
Hbeta1 = spectrumMean1[betaBegin:betaEnd]

Halpha2 = spectrumMean2[alphaBegin:alphaEnd]
Hbeta2 = spectrumMean2[betaBegin:betaEnd]

initialCont = 10
initialSigma = 1

result_alpha1 = gaussian_model.fit(Halpha1, x = HalphaLambda, cont = initialCont, A = (np.max(Halpha1) - initialCont), sigma = initialSigma, x0 = lambdas[int(alphaBegin / 2 + alphaEnd / 2)])
alpha1Cont = result_alpha1.params['cont'].value
alpha1A = result_alpha1.params['A'].value
alpha1Sigma = result_alpha1.params['sigma'].value
alpha1X0= result_alpha1.params['x0'].value
print(result_alpha1.fit_report())

result_alpha2 = gaussian_model.fit(Halpha2, x = HalphaLambda, cont = initialCont, A = (np.max(Halpha2) - initialCont), sigma = initialSigma, x0 = lambdas[int(alphaBegin / 2 + alphaEnd / 2)])
alpha2Cont = result_alpha2.params['cont'].value
alpha2A = result_alpha2.params['A'].value
alpha2Sigma = result_alpha2.params['sigma'].value
alpha2X0= result_alpha2.params['x0'].value
print(result_alpha2.fit_report())

result_beta1 = gaussian_model.fit(Hbeta1, x = HbetaLambda, cont = initialCont, A = (np.max(Hbeta1) - initialCont), sigma = initialSigma, x0 = lambdas[int(betaBegin / 2 + betaEnd / 2)])
beta1Cont = result_beta1.params['cont'].value
beta1A = result_beta1.params['A'].value
beta1Sigma = result_beta1.params['sigma'].value
beta1X0= result_beta1.params['x0'].value
print(result_beta1.fit_report())

result_beta2 = gaussian_model.fit(Hbeta2, x = HbetaLambda, cont = initialCont, A = (np.max(Hbeta2) - initialCont), sigma = initialSigma, x0 = lambdas[int(betaBegin / 2 + betaEnd / 2)])
beta2Cont = result_beta2.params['cont'].value
beta2A = result_beta2.params['A'].value
beta2Sigma = result_beta2.params['sigma'].value
beta2X0= result_beta2.params['x0'].value
print(result_beta2.fit_report())

fig, axes = plt.subplots(2, 2)

alphaX = np.linspace(lambdas[alphaBegin], lambdas[alphaEnd], 100)
betaX = np.linspace(lambdas[betaBegin], lambdas[betaEnd], 100)

axes[0, 0].plot(HalphaLambda, Halpha1, label = 'H' + r'$_{\alpha}$' + ' galaxy 1')
axes[0, 0].plot(alphaX, gaussian(alphaX, alpha1Cont, alpha1A, alpha1Sigma, alpha1X0), 'k--', label = 'best fit')
axes[0, 0].set_xlabel(r'$\lambda, A$')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].grid()
axes[0, 0].legend()

axes[1, 0].plot(HalphaLambda, Halpha2, label = 'H' + r'$_{\alpha}$' + ' galaxy 2')
axes[1, 0].plot(alphaX, gaussian(alphaX, alpha2Cont, alpha2A, alpha2Sigma, alpha2X0), 'k--', label = 'best fit')
axes[1, 0].set_xlabel(r'$\lambda, A$')
axes[1, 0].set_ylabel('Intensity')
axes[1, 0].grid()
axes[1, 0].legend()

axes[0, 1].plot(HbetaLambda, Hbeta1, label = 'H' + r'$_{\beta}$' + ' galaxy 1')
axes[0, 1].plot(betaX, gaussian(betaX, beta1Cont, beta1A, beta1Sigma, beta1X0), 'k--', label = 'best fit')
axes[0, 1].set_xlabel(r'$\lambda, A$')
axes[0, 1].set_ylabel('Intensity')
axes[0, 1].grid()
axes[0, 1].legend()

axes[1, 1].plot(HbetaLambda, Hbeta2, label = 'H' + r'$_{\beta}$' + ' galaxy 2')
axes[1, 1].plot(betaX, gaussian(betaX, beta2Cont, beta2A, beta2Sigma, beta2X0), 'k--', label = 'best fit')
axes[1, 1].set_xlabel(r'$\lambda, A$')
axes[1, 1].set_ylabel('Intensity')
axes[1, 1].grid()
axes[1, 1].legend()

plt.show()

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










