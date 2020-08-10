# FFD
How to generate a stellar Flare Frequency Distribution, and its uncertainties

<img src="https://github.com/jradavenport/FFD/blob/master/ffd.png" alt="ffd package example" width="400"/>


## Example: Basic

A basic example of how to make the cumulative Flare Frequency Distribution plot, with both contours from the Gaussian kernel density estimation, and the standard scatter plot. You should already have detected your flares and computed their durations (used in the S/N estimation of the event energies) and the equivalent durations (integral of the flare in zero-registered relative flux).


````python
from FFD import FFD, FlareKernel

x,y,xe,ye = FFD(EquivDur, dur=Tstop-Tstart, Lum=30.35, TotDur=50.4,
                fluxerr=np.median(fluxerr)/np.median(flux))

im, xx, yy = FlareKernel(x,y,xe,ye)

plt.contour(xx, yy, im)
plt.errorbar(x,y, xerr=xe, yerr=ye)
plt.xlabel('log Energy (erg)')
plt.ylabel('log Flare Rate (day$^{-1}$)')
````


## Example: Better Error Bars and Power Law Fit

If we have error estimates for the equivalent durations, we can use these to better estimate the x errors on the FFD.

````python
import numpy as np
from FFD import FFD, FFD_powerlaw FlareKernel

x,y,xe,ye = FFD(EquivDur, edErr=EDerr, Lum=30.35, TOTEXP=50.4)

im, xx, yy = FlareKernel(x,y,xe,ye)

plt.contour(xx, yy, im)
plt.errorbar(x,y, xerr=xe, yerr=ye)
plt.xlabel('log Energy (erg)')
plt.ylabel('log Flare Rate (day$^{-1}$)')
````

We can then fit a power law to the data and estimate the energy below which the data is incomplete.

````python
b0, b1, b0_err, b1_err, cutoff = FFD_powerlaw(x, y, xe, ye, findXmin=True)

xmodel = np.linspace(np.min(x), np.max(x))
ymodel = b0*xmodel + b1

plt.plot(xmodel, ymodel)
plt.axvlines(cutoff, linestyle='--')
````
