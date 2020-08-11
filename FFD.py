import numpy as np
from astropy.modeling.models import Gaussian2D
from scipy.odr import ODR, Model, Data
from scipy import stats

def FFD(ED, edErr=[], TOTEXP=1., Lum=30., fluxerr=0., dur=[], logY=True, est_comp=False):
    '''
    Given a set of stellar flares, with accompanying durations light curve properties,
    compute the reverse cumulative Flare Frequency Distribution (FFD), and
    approximate uncertainties in both energy and rate (X,Y) dimensions.
    This diagram can be read as measuring the number of flares per day at a
    given energy or larger.
    Not a complicated task, just tedious.
    Y-errors (rate) are computed using Poisson upper-limit approximation from
    Gehrels (1986) "Confidence limits for small numbers of events in astrophysical data", https://doi.org/10.1086/164079
    Eqn 7, assuming S=1.
    X-errors (event energy) are computed following Signal-to-Noise approach commonly
    used for Equivalent Widths in spectroscopy, from
    Vollmann & Eversberg (2006) "Astronomische Nachrichten, Vol.327, Issue 9, p.862", https://dx.doi.org/10.1002/asna.200610645
    Eqn 6.
    Parameters
    ----------
    ED : array of Equiv Dur's, need to include a luminosity!
    edErr : Error estimates for ED measurements
    TOTEXP : total duration of observations, in days
    Lum : the log luminosity of the star (pass as an array if considering multiple stars)
    fluxerr : the average flux errors of your data in relative flux units! (only needed if edErr not provided)
    dur : array of flare durations (to estimate x errors if ED errors not provided)
    logY : if True return Y-axis (and error) in log rate (Default: True)
    est_comp : estimate incompleteness using histogram method, scale Y errors?
        (Default: True)
    Returns
    -------
    ffd_x, ffd_y, ffd_xerr, ffd_yerr
    X coordinate always assumed to be log_10(Energy)
    Y coordinate is log_10(N/Day) by default, but optionally is N/Day
    Upgrade Ideas
    -------------
    - More graceful behavior if only an array of flares and a total duration are
        specified (i.e. just enough to make ffd_x, ffd_y)
    - Better propogation of specific flux errors in the light curve, rather than
        average error used
    - Include detrending errors? (e.g. from a GP)
    - Asymmetric Poisson errors?
    - Better handling of incompleteness?
    '''
    
    energy = ED*Lum
    
    # REVERSE sort the flares in energy
    ss = np.argsort(energy)[::-1]
    ffd_x = np.log10(energy[ss])

    Num = np.arange(1, len(ffd_x)+1)
    ffd_y = Num / TOTEXP

    # approximate the Poisson Y errors using Gehrels (1986) eqn 7
    Perror = np.sqrt(Num + 0.75) + 1.0
    ffd_yerr = Perror / TOTEXP

    # estimate completeness using the cumulative distribution of the histogram
    if est_comp:
        # make very loose guess at how many bins to choose
        nbin = int(np.sqrt(len(ffd_x)))
        if nbin < 10:
            nbin=10 # but use at least 10 bins

        # make histogram of the log(energies)
        hh, be = np.histogram(ffd_x, bins=nbin, range=[np.nanmin(ffd_x), np.nanmax(ffd_x)])
        hh = hh/np.nanmax(hh)
        # make cumulative distribution of the histogram, scale to =1 at the hist peak
        cc = np.cumsum(hh)/np.sum(hh[0:np.argmax(hh)])
        be = (be[1:]+be[0:-1])/2
        # make completeness = 1 for energies above the histogram peak
        cc[np.argmax(hh):] = 1
        # interpolate the cumulative histogram curve back to the original energies
        ycomp = np.interp(ffd_x, be, cc)
        # scale the y-errors by the completeness factor (i.e. inflate small energy errors)
        ffd_yerr = ffd_yerr / ycomp

    if logY:
        # transform FFD Y and Y Error into log10
        ffd_yerr = np.abs(ffd_yerr / np.log(10.) / ffd_y)
        ffd_y = np.log10(ffd_y)

    # compute X uncertainties for FFD
    # Estimate from flare durations and average flux errors if ED error bars not provided
    if (len(edErr)==len(ffd_x)):
        S2N = ED / np.sqrt(ED + edErr)
        ffd_xerr = np.abs(edErr / np.log(10.) / ED[ss])
    elif len(dur)==len(ffd_x):
        # assume relative flux error = 1/SN
        S2N = 1/fluxerr
        # based on Equivalent Width error
        # Eqn 6, Vollmann & Eversberg (2006) Astronomische Nachrichten, Vol.327, Issue 9, p.862
        ED_err = np.sqrt(2)*(dur[ss]*86400. - ED[ss])/S2N
        ffd_xerr = np.abs((ED_err) / np.log(10.) / ED[ss]) # convert to log
    else:
        # not particularly meaningful, but an interesting shape. NOT reccomended
        print('Warning: No flare durations or ED error estimates provided. Making bad assumptions\
               about the FFD X Error!')
        ffd_xerr = (1/np.sqrt(ffd_x-np.nanmin(xT))/(np.nanmax(ffd_x)-np.nanmin(ffd_x)))

    return ffd_x, ffd_y, ffd_xerr, ffd_yerr

def FFD_powerlaw(logx, logy, logxerr, logyerr, findXmin=False):
    '''
    Given a FFD, provide the best fit parameters for a power law function. This is done using
    orthogonal distance regression, so that both x and y errors can be accounted for. As a crude
    way to account for incompleteness, determine a minimum energy, below which the fit gets bad. 
    This is done by iteratively fitting with different values for xmin, and finding the value of
    xmin that minimizes the KS distance between the model and the data. See Clauset 2007
    (arXiv:0706.1062), sec 3.3 for an explanation of the algorithm.
    Parameters
    ----------
    logx : The log-scaled x values
    logy : The log-scaled y values
    logxerr : The log-scaled x error bars
    logyerr : The log-scaled y error bars
    findXmin : Iteratively determine the best x value below which to trim the data
    Returns
    -------
    b0, b1, b0_err, b1_err, cutoff
    Best fit parameters for the power law slope and normalization, along with the optimal xmin
    '''
    def f(B, x):
        if B[0] > 0:
            return np.inf
        return B[0]*x + B[1]
    
    if findXmin:
        cutoff_vals = np.linspace(np.min(logx), np.max(logx))
    else:
        cutoff_vals = -np.inf
        
    ks_vals = np.zeros_like(cutoff_vals)
    param_arr = np.empty((len(ks_vals), 4))
    
    for idx, e_cut in enumerate(cutoff_vals):
        # Initial guess for powerlaw fit
        b00, b10 = -0.5, 10

        mask = logx > e_cut
        
        # Make the KS distance large if we threw out all of the data
        if len(logx[mask]) < 1:
            ks_vals[idx] = np.inf
            continue

        linear = Model(f)
        mydata = Data(logx[mask], logy[mask], wd=1/logxerr[mask]**2, we=1/logyerr[mask]**2)
        myodr = ODR(mydata, linear, beta0=[b00, b10])
        myoutput = myodr.run()
        b0, b1 = myoutput.beta[0], myoutput.beta[1]
        b0_err, b1_err = myoutput.sd_beta[0], myoutput.sd_beta[1]

        # Record the KS distance and best fit parameters for this particular xmin value
        ks_vals[idx] = stats.ks_2samp(logy[mask], b0*logx[mask] + b1).statistic
        param_arr[idx][0] = b0
        param_arr[idx][1] = b1
        param_arr[idx][2] = b0_err
        param_arr[idx][3] = b1_err
        
    # Use the parameters that correspond to the minimum KS distance
    best_ks_idx = np.argmin(ks_vals)
    b0 = param_arr[best_ks_idx][0]
    b1 = param_arr[best_ks_idx][1]
    b0_err = param_arr[best_ks_idx][2]
    b1_err = param_arr[best_ks_idx][3]
    cutoff = cutoff_vals[best_ks_idx]
    
    return b0, b1, b0_err, b1_err, cutoff

def FlareKernel(x, y, xe, ye, Nx=100, Ny=100, xlim=[], ylim=[], return_axis=True):
    '''
    Use 2D Gaussians (from astropy models) to make a basic kernel density,
    with errors in both X and Y considered. Turn into a 2D "image"
    Upgrade Ideas
    -------------
    It's slow. Since Gaussians are defined analytically, maybe this could be
    re-cast as a single array math opperation, and then refactored to have the
    same fit/evaluate behavior as KDE functions.  Hmm...
    '''

    if len(xlim) == 0:
        xlim = [np.nanmin(x) - np.nanmean(xe), np.nanmax(x) + np.nanmean(xe)]
    if len(ylim) == 0:
        ylim = [np.nanmin(y) - np.nanmean(ye), np.nanmax(y) + np.nanmean(ye)]

    xx,yy = np.meshgrid(np.linspace(xlim[0], xlim[1], Nx),
                        np.linspace(ylim[0], ylim[1], Ny), indexing='xy')
    dx = (np.max(xlim)-np.min(xlim)) / (Nx-1)
    dy = (np.max(ylim)-np.min(ylim)) / (Ny-1)

    im = np.zeros_like(xx)

    for k in range(len(x)):
        g = Gaussian2D(amplitude=1/(2*np.pi*(xe[k]+dx)*(ye[k]+dy)),
                       x_mean=x[k], y_mean=y[k], x_stddev=xe[k]+dx, y_stddev=ye[k]+dy)
        tmp = g(xx,yy)
        if np.isfinite(np.sum(tmp)):
            im = im + tmp

    if return_axis:
        return im, xx, yy
    else:
        return im
