# FFD
How to generate a stellar Flare Frequency Distribution, and its uncertainties

<img src="https://github.com/jradavenport/FFD/blob/master/ffd.png" alt="ffd package example" width="400"/>


## Example

A basic example of how to make the cumulative Flare Frequency Distribution plot, with both contours from the Gaussian kernel density estimation, and the standard scatter plot. You should already have detected your flares and computed their durations (used in the S/N estimation of the event energies) and the equivalent durations (integral of the flare in zero-registered relative flux).


````python
from FFD import ffd, FlareKernel

x,y,xe,ye = FFD(EquivDur, dur=Tstop-Tstart, Lum=30.35, TotDur=50.4,
                fluxerr=np.median(fluxerr)/np.median(flux))

im, xx, yy = FlareKernel(x,y,xe,ye)

plt.contour(xx, yy, im)
plt.errorbar(x,y, xerr=xe, yerr=ye)
plt.xlabel('log Energy (erg)')
plt.ylabel('log Flare Rate (day$^{-1}$)')
````
