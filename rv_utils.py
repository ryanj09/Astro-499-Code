
import numpy as N
import astropy.io.fits as pyfits
from glob import glob
import os
from scipy.io import readsav
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb

from astropy import constants as const
import astropy.coordinates as coord
import astropy.units as u
from BaryTime import BaryTime

def phase(time, period, T0):
    return ((time - T0) / period) % 1.

def spectrum(specfile):
    hdulist = pyfits.open(specfile)
    hdr = hdulist[0].header

    crpix1 = hdr['CRPIX1']
    crval1 = hdr['CRVAL1']
    cd1_1 = hdr['CDELT1']

    flux = hdulist[0].data
    spec_length = len(flux)
    wavelength = cd1_1 * (N.arange(spec_length) - (crpix1-1)) + crval1

    return wavelength, flux

def avg_snr(wavelength, flux, err, range=[4550,4700]):

    w = (wavelength >= range[0]) & (wavelength <= range[1])
    return N.mean(flux[w]/err[w])
    

def rename_specs(directory,side='blue'):
    """rename files from blueNNNN_flux.spec.fits to 
    blue0054_YYMMDD_flux.spec.fits to avoid collisions; also err"""

    absdir = os.path.abspath(directory) + '/'

    pattern = side+'????_flux.spec.fits'

    for file in glob(absdir+pattern):
        hdulist = pyfits.open(file)
        hdr = hdulist[0].header
        datestr = hdr['DATE-OBS']
        shortdate = datestr[0:4]+datestr[5:7]+datestr[8:10]

        orig_filename = file.split('/')[-1]
        if orig_filename.startswith('blue'):
            new_filename = orig_filename[:8]+'_'+shortdate + orig_filename[8:]
        elif orig_filename.startswith('red'):
            new_filename = orig_filename[:7]+'_'+shortdate + orig_filename[7:]
        else:
            raise ValueError("Unknown input filename")

        os.rename(absdir+orig_filename,absdir+new_filename)
        os.rename(absdir+orig_filename.replace('spec','err'),
            absdir+new_filename.replace('spec','err'))


def make_speclist(directory,pattern='blue*flux.spec.fits',
    outfile='speclist.dat',
    K=None, P=None, rvobsfile=None, range=[4550,4700]):
    # if K (km/s) and P (hours) are specified, calculate vblur
    # if rv.obs file is given, use fit values as a guess

    
    outstr = """# mjd exptime velguess velblur seeing snr path/to/image/root
# for example, remove the '.spec.txt' from filenames
# SNR calculation: avg_snr, range=[{},{}]
# using flux/error as given by iraf error bars\n""".format(range[0],range[1])

    absdir = os.path.abspath(directory) + '/'
    if rvobsfile is not None:
        with open(rvobsfile,'r') as f:
            rvlines = f.readlines()

    lines = []
    mjds = []
    for file in glob(absdir+pattern):
        hdulist = pyfits.open(file)
        hdr = hdulist[0].header
        mjd = hdr['MJD']
        exptime = hdr['EXPTIME']
        if rvobsfile is not None:
            obs_mid = find_midpoint_time(hdr)
            mid_str = datetime.datetime.strftime(obs_mid,'%Y %m %d %H:%M:%S')
            velguess = 0
            for line in rvlines:
                if line.startswith(mid_str):
                    velguess = float(line.split(' ')[-1])
        else:
            velguess = 0
        if (K is not None) and (P is not None):
            raise ValueError("Double check velblur calculation")
            velblur = K * N.sin(2*N.pi*exptime/(P*3600.))
        else:
            velblur = 0
        seeing = hdr['FWHM']
        wavelength, flux = spectrum(file)
        _ , err = spectrum(file.replace('.spec.fits','.err.fits'))
        snr = avg_snr(wavelength, flux, err, range=range)
        lines.append('{} {} {:.1f} {:.1f} {} {:.1f} {}'.format(mjd, exptime, 
            velguess, velblur, seeing, snr, file.replace('.spec.fits','')))
        mjds.append(mjd)

    lines = [line for (line,mjd) in sorted(zip(lines,mjds))]
    outfile = open(absdir+outfile,'w')
    outfile.write(outstr + '\n'.join(lines))
    outfile.close()
    
def find_midpoint_time(hdr):
    utshut = hdr['UTSHUT']
    obs_start_datetime = datetime.datetime.strptime(utshut,
        '%Y-%m-%dT%H:%M:%S.%f')
    exptime = hdr['EXPTIME']
    obs_mid = obs_start_datetime + datetime.timedelta(0,exptime/2.)
    return obs_mid
    
def list_best_template(savfile='bestfitvels.sav'):
    """take output of getvel.pro and print best templates"""

    sav = readsav(savfile, python_dict=True)
    minchi2 = sav['minchi2fit']
    fitvels = sav['fitvels']
    velerrs = sav['velerrs']
    observations = sav['observations']

    with open(sav['temfile'],'r') as f:
        templates = N.array(f.readlines())

    if len(minchi2.shape) == 2:
        (ntemplates, nobs) = minchi2.shape
        wbest = minchi2.argmin(axis=0)
        best_vel = fitvels[wbest,N.arange(nobs)]
        best_err = velerrs[wbest,N.arange(nobs)]
        best_templates = templates[wbest]
    else:
        # only one template per obs
        nobs = minchi2.shape[0]
        ntemplates = 1
        best_vel = fitvels
        best_err = velerrs
        best_templates = N.chararray(nobs)
        best_templates[:] = templates[0]


    for i, obs in enumerate(observations):
        print('{} {} {:.1f} +/- {:.1f}'.format(
            # temp only
            #obs.split('/')[-1], best_templates[i].split('/')[-1].split('G')[0],
            obs.split('/')[-1], best_templates[i].split('/')[-1][:-1],
            best_vel[i], best_err[i]))




def correct_rvs_homebrew(savfile='bestfitvels.sav',outfile='test_rv_vlsr.txt',
    observatory='Palomar'):
    """take output of getvel.pro and correct to solar system barycenter
    
    requires all observations to come from the same observatory"""


    sav = readsav(savfile, python_dict=True)
    minchi2 = sav['minchi2fit']
    fitvels = sav['fitvels']
    velerrs = sav['velerrs']
    exptimes = sav['exptime']
    if len(minchi2.shape) == 2:
        (ntemplates, nobs) = minchi2.shape
        wbest = minchi2.argmin(axis=0)
        best_vel = fitvels[wbest,N.arange(nobs)]
        best_err = velerrs[wbest,N.arange(nobs)]
    else:
        nobs = minchi2.shape[0]
        ntemplates = 1
        best_vel = fitvels
        best_err = velerrs

    #intermediate_file = 'rv.obs'
    #f = open(intermediate_file,'w')

    f = open(outfile,'w')
    f.write("# BJD_TDB EXPTIME VLSR VERR VSYS\n")

    sys_errs = []
    for i,obsfile in enumerate(sav['observations'].tolist()):

        velocity = best_vel[i]
        if observatory == 'Palomar':

            # DBSP spectra reduced by my pipeline
            hdulist = pyfits.open(obsfile+'.spec.fits')
            hdr = hdulist[0].header
            ra = hdr['RA']
            dec = hdr['DEC']
            sys_velerr = hdr['VERR'] # velocity error do to wavelength uncertainty
            sys_errs.append(sys_velerr)

            obs_mid = find_midpoint_time(hdr)

            tel_loc = coord.EarthLocation(lat=coord.Latitude('33d21m21.6s'),
                                       lon=coord.Longitude('-116d51m46.80s'),
                                                              height=1706.)

            t = BaryTime(obs_mid, format='datetime',scale='utc',
                location=tel_loc)

        elif observatory == 'Keck':

            # LRIS spectra reducd by Dan Perley's pipeline
            sys_errs.append(0.)
            rahms = get_dperley_header(obsfile+'.spec','RA').strip().strip("'")
            decdms = get_dperley_header(obsfile+'.spec','DEC').strip().strip("'")
            c = coord.SkyCoord(ra=rahms,dec=decdms,unit=(u.hourangle, u.deg))

            ra = c.ra.value
            dec = c.dec.value

            exptime = get_dperley_header(obsfile+'.spec','EXPTIME',
                converter = N.float) * u.second
            mjd = get_dperley_header(obsfile+'.spec','MJD-OBS',
                converter = N.float) * u.day

            
            
            tel_loc = coord.EarthLocation(lat=coord.Latitude('19d49m34.9s'),
                                       lon=coord.Longitude('-155d28m30.04s'),
                                                              height=4145.)

            t = BaryTime(mjd+exptime/2., format='mjd',scale='utc',
                location=tel_loc)

        else:
            raise NotImplementedError('Observatory {} not implemented'.format(
                observatory))

        PSR_coords = coord.SkyCoord(ra,dec,frame='icrs', unit=u.deg)

        (h_pos,h_vel,b_pos,b_vel) = t._obs_pos()
        m, vect = t._vect(PSR_coords)
        v_corr = (b_vel.dot(vect)*const.au/(1 * u.day)).to(u.km/u.s)
        bjd_tdb = t.bcor(PSR_coords).jd

        f.write('{} {} {} {:.2f} {}\n'.format(bjd_tdb[0], exptimes[i], 
            best_vel[i]+v_corr.value[0], best_err[i], sys_errs[i]))
    f.close()

def get_dperley_header(filename, keyword, converter=lambda x:x):
    
    for line in open(filename,'r'):
        if line.startswith('# {}'.format(keyword)):
            return converter(line.split('=')[-1][1:])

def correct_rvs(savfile='bestfitvels.sav',outfile='rv_vlsr.txt',
    observatory='Palomar'):
    """take output of getvel.pro and correct to solar system barycenter
    
    requires all observations to come from the same observatory"""

    sav = readsav(savfile, python_dict=True)
    minchi2 = sav['minchi2fit']
    fitvels = sav['fitvels']
    velerrs = sav['velerrs']
    exptimes = sav['exptime']
    (ntemplates, nobs) = minchi2.shape

    wbest = minchi2.argmin(axis=0)
    best_vel = fitvels[wbest,N.arange(nobs)]
    best_err = velerrs[wbest,N.arange(nobs)]

    intermediate_file = 'rv.obs'
    f = open(intermediate_file,'w')

    sys_errs = []
    for i,obsfile in enumerate(sav['observations'].tolist()):
        hdulist = pyfits.open(obsfile+'.spec.fits')
        hdr = hdulist[0].header
        ra = hdr['RA']
        dec = hdr['DEC']
        sys_velerr = hdr['VERR'] # velocity error do to wavelength uncertainty
        sys_errs.append(sys_velerr)

        obs_mid = find_midpoint_time(hdr)
        f.write('{} {} {} 0 {:.3f}\n'.format(
            datetime.datetime.strftime(obs_mid,'%Y %m %d %H:%M:%S'), ra, dec, 
            best_vel[i]))
    f.close()
        
    hjds, vlsrs = rvcorrect(observatory=observatory)

    f = open(outfile,'w')
    f.write("# HJD EXPTIME VLSR VERR VSYS\n")
    for i in range(nobs):
        f.write('{} {} {} {:.2f} {}\n'.format(hjds[i], exptimes[i], vlsrs[i],
            best_err[i], sys_errs[i]))
    f.close()

def quick_list_rvs(savfile='bestfitvels.sav'):
    """take output of getvel.pro and print best-fit velocity (not barycentered, 
        no systematic velocity error)"""

    sav = readsav(savfile, python_dict=True)
    minchi2 = sav['minchi2fit']
    fitvels = sav['fitvels']
    velerrs = sav['velerrs']
    exptimes = sav['exptime']
    (ntemplates, nobs) = minchi2.shape

    wbest = minchi2.argmin(axis=0)
    best_vel = fitvels[wbest,N.arange(nobs)]
    best_err = velerrs[wbest,N.arange(nobs)]
    for i,obsfile in enumerate(sav['observations'].tolist()):
        print('{}\t{:.1f}\t+/-\t{:.1f}'.format(obsfile.split('/')[-1], best_vel[i], best_err[i]))


def rvcorrect(infile='rv.obs',observatory='Palomar'):
    from pyraf import iraf
    iraf.astutil(_doprint=0)
    iraf.unlearn('rvcorrect')

    iraf.rvcorrect.files = infile
    iraf.rvcorrect.observatory = observatory
    corr = iraf.rvcorrect(Stdout=1)

    hjds = []
    vlsrs = []
    for line in corr:
        if not line.startswith('#'):
            tok = line.split()
            hjds.append(tok[0])
            vlsrs.append(tok[3])

    return hjds, vlsrs


def fold_rvs(period_days, rvfile='rv_vlsr.txt', tasc_mjd=None):
    import matplotlib.pyplot as plt

    dat = N.genfromtxt(rvfile, names=['hjd','exptime','rv','verr','vsys'])
    mjd = dat['hjd'] - 2400000.5
    err = N.sqrt(dat['verr']**2. + dat['vsys']**2.)

    phase = mjd_to_phase(mjd, period_days, tasc_mjd = tasc_mjd)

    plt.errorbar(phase,dat['rv'],xerr=dat['exptime']/2./(3600*24.),
        yerr=err,fmt='.')
    plt.xlim(0,1)
    plt.xlabel('Phase')
    plt.ylabel('Radial Velocity (km/s)')

def mjd_to_phase(mjd, period_days, tasc_mjd=None):

    if tasc_mjd is None:
        # if we don't know the mjd, choose an arbitrary one for now
        tasc_mjd = mjd[0]

    phase = ((mjd-tasc_mjd)/period_days) % 1
    return phase


#def ptf_lightcurve_to_nightfall()

def rv_estimate(spec1, spec2, step_range=[-200, 200], step_size = 1,
    wavelength_range = [3000,10000]):
    """Quick and dirty estimate of the RV shift between two spectra.
    Because this is sensitive to the fluxing, for now it's only recommended
    for data taken on one night.

    spec1, spec2: fits filenames ("*_flux.spec.fits")

    TODOS: interpolate to common grid, do gentle flux correction to 
        allow for intra-night comparisons
    wrap with a routine that folds at period and tries to fit sinusoid"""

    


# j2129:
#stack_plot_phase(glob('red*spec.fits'), .3176*2.,tasc_mjd=55702.111169,
#    semiamplitude=255.5,mean_anomaly=-62.9,xlim=[5700,8900], color='crimson')

# warning--hard-coded barycentering for J2129 right now
def stack_plot_phase(spec_list, period_days, tasc_mjd=None, 
    semiamplitude=None, mean_anomaly=None,
    alpha=.7, yheight=0.1, xlim = None, ylim = None, color=None,
    optical_phase = False):
    """Plot several spectra on top of each other with matplotlib.
    Consider also iraf.specplot('spec1,spec2,spec3').

    Parameters
    ----------
    spec_list : list of strings
        List of text or fits spectra extracted by extract1D
        (e.g., ['red0001_flux.spec.txt',red0002_flux.spec.fits'])
    alpha : float, default 1.0
        Opacity of the plot lines
    yheight : float, default 0.1
        Proportion of phase 0-1 to scale the range of each spectrum to.
    optical_phase : boolean, default False
        if False, use TASC as phase zero point; else use TASC-0.25
    """

    category20= N.array(['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b',
    '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22',
    '#dbdb8d', '#17becf', '#9edae5'])

    sb.set_style('white')
#    sb.set_palette(sb.dark_palette(color, n_colors = len(spec_list)),
#        n_colors=len(spec_list))
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['legend.fontsize'] = 14


    atm_bands = [
        [6701, 6715], # dome overhead lamp flourescence feature
        [6860,7000],
        [7570,7700],
        [7150,7350],
        [8100,8401]] # trim a funky bin with +1

    def exclude_bands(wave, bands):
        winband = N.zeros(len(wave)).astype(N.bool)
        for band in bands:
            winband |= ((wave >= band[0]) & (wave <= band[1]))
        #woutband = ~winband
        return winband


    phases = []
    xs = []
    ys = []
    labels = []
    mjds = []
            


    for spec in spec_list:
        if spec.endswith('txt'):
            dat = N.genfromtxt(spec, names='wave, flux',
                dtype='f4, f4')
            raise ValueError('Need fits spectra')
        elif spec.endswith('fits'):
            hdulist = pyfits.open(spec)
            hdr = hdulist[0].header
            flux = hdulist[0].data
            crpix1 = hdr['CRPIX1']
            crval1 = hdr['CRVAL1']
            cd1_1 = hdr['CDELT1']
            mjd = hdr['MJD']
            spec_length = len(flux)
            wave = cd1_1 * (N.arange(spec_length) - (crpix1-1)) + crval1
            dat = {'wave':wave,'flux':flux}

        P200_loc = coord.EarthLocation(lat=coord.Latitude('33d21m21.6s'),
            lon=coord.Longitude('-116d51m46.80s'),
            height=1706.)
        t = BaryTime(mjd, format='mjd',scale='utc', location=P200_loc)

        PSR_coords = coord.SkyCoord(322.43776,-4.48521,frame='icrs', unit=u.deg)
        bjd_tdb = t.bcor(PSR_coords).mjd

        phase = mjd_to_phase(bjd_tdb, period_days, tasc_mjd=tasc_mjd)[0]


        # for j2129: convert to optical phase convention
        tasc_phase = phase
        if optical_phase:
            phase = (phase - 0.25 ) % 1

        
        if xlim is None:
            wmin = 0
        else:
            wmin = N.nonzero(wave > xlim[0])[0][0]
        offset_val  = N.median(dat['flux'][wmin:wmin+100]) 
        pct = N.percentile(dat['flux'],[5,95])
        scale_val = yheight/(pct[1]-pct[0])
        w = exclude_bands(wave, atm_bands)
        dat['flux'][w] = N.NaN
        # correct for the binary shift
        if (semiamplitude is not None) and (mean_anomaly is not None):
            predicted_shift = mean_anomaly -  semiamplitude * \
                N.cos(2.*N.pi * tasc_phase) 
            # delta_lambda/lambda_emitted = v/c in nonrel. case
            # so lambda_emitted = lambda_obs/ (1+v/c)
            dat['wave'] = dat['wave'] / (1 + predicted_shift/299792.458)
        mjds.append(mjd)
        phases.append(phase)
        xs.append(dat['wave'])
        ys.append((dat['flux']-offset_val)*scale_val+phase)
        labels.append(spec)

    mjds_set = N.sort(N.unique(N.floor(mjds)))
    assert(len(mjds_set) < len(category20))
    if color is None:
        colors = category20[[N.where(mjds_set == N.floor(mi))[0][0] for mi in mjds]]
    else:
        colors = [color for mi in mjds]

    fig = plt.figure(figsize=(20,8))
    ax = plt.subplot(111)
    args = N.argsort(phases)
    for i in args:
        print(labels[i], phases[i])
        plt.plot(xs[i], ys[i], label = labels[i], alpha=alpha, linewidth=1.,
            color=colors[i])
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Orbital Phase (cycles)")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    #plt.legend()
    sb.despine()
    plt.show()
    return ax

def label_lines(line_list, ax=None, label_y = 1., 
    line_top = 0.99, line_mid = 0.96, line_bottom = 0.93, color='black'):
    # input is a tuple of tuples: each tuple is the label and a tuple with 
    #    one or more wavelengths
    # e.g.,  ( ('H$\alpha$',(6562.8)), ('Ca II',(8498.03, 8542.09, 8662.14)) )

    for label, lines in line_list:
        
        xmid = N.mean(lines)
        ax.text(xmid, label_y, label,
            horizontalalignment='center', verticalalignment='bottom',
            fontsize='small', rotation='vertical')

        if N.array(lines).size == 1:
            ax.plot([xmid,xmid],[line_bottom,line_top],color=color)
        else:
            ax.plot([xmid,xmid],[line_mid,line_top],color=color)
            ax.plot([N.min(lines),N.max(lines)],[line_mid,line_mid],color=color)
            for line in lines:
                ax.plot([line,line],[line_bottom,line_mid],color=color)
            
    
    plt.show()
