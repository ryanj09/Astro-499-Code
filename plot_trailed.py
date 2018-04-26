
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c
from glob import glob
from scipy.interpolate import interp1d
from astropy.time import Time
import pyspeckit as psk
import seaborn as sb
import warnings

from rv_utils import label_lines
import seaborn as sb
sb.set_style('ticks')


# pars = main()
# build_pyspeckit(pars)
# loop_fit(pars)
# df = read_fits(...)

def main():
    #base_dir = '/lustre/PTF/observation_data/variable/LRIS_20161102/spredux1d/'
    base_dir = 'data/'

    blue_files = np.sort(glob(base_dir + 'celsb*PTFS1623al.spec')).tolist()
    red_files = np.sort(glob(base_dir + 'celsr*PTFS1623al.spec')).tolist()

    pars = {'blue':{'wmin':3100,'wmax':5600,'dlambda':1.25},
                 'red':{'wmin':6420,'wmax':8880,'dlambda':1.20}}

    for side in ['blue','red']:
        pars[side]['nbins'] = int((pars[side]['wmax'] - 
                                pars[side]['wmin'])/pars[side]['dlambda'])
        pars[side]['wavelengths'] = np.linspace(pars[side]['wmin'],
                                                pars[side]['wmax'],
                                                pars[side]['nbins'])


    pars['blue']['files'] = blue_files
    pars['red']['files'] = red_files 


    for side in ['blue','red']:
        pars[side]['raw_data'] = []
        pars[side]['raw_error'] = []
        pars[side]['raw_wavelength'] = []
        pars[side]['start_time'] = []
        pars[side]['end_time'] = []
        pars[side]['header'] = []

        # interpolated onto common grid
        pars[side]['data'] = np.zeros([len(pars[side]['files']),
                                        pars[side]['nbins']])
        pars[side]['median_data'] = np.zeros([len(pars[side]['files']),
                                        pars[side]['nbins']])
    
    for side in ['blue','red']:
        for i, fname in enumerate(pars[side]['files']):
            df, header = read_lpipe_spec(fname)
            pars[side]['start_time'].append(Time(header['DATE_BEG']).mjd)
            pars[side]['end_time'].append(Time(header['DATE_END']).mjd)
            pars[side]['raw_data'].append(df.flux.values)
            pars[side]['raw_error'].append(df.flux_unc.values)
            pars[side]['raw_wavelength'].append(df.wavelength.values)
            pars[side]['header'].append(header)


            flux = interpolate_spec(df,pars[side]['wavelengths'])
            pars[side]['data'][i,:] = flux
            pars[side]['median_data'][i,:] = flux/np.median(flux[np.isfinite(flux)])

                # also think about a log cmap norm?
                # and make a phase-binned coadd

    return pars


def read_lpipe_spec(fname):

    header = {}
    with open(fname,'r') as f:
        for line in f:
            if line.startswith('#'):
                line = line.replace("'","")
                tok = line[2:].split('=')
                if (len(tok) == 2):
                    header[tok[0].strip()] = tok[1].split('/')[0].strip("\'").strip()
    
    #wavelength     flux         sky_flux     flux_unc     xpixel  ypixel  response
    df = pd.read_table(fname,sep='\s+',comment='#',
        names =  ['wavelength','flux','sky_flux','flux_unc','xpixel','ypixel',
        'response'])

    return df, header


def interpolate_spec(df,target_waves):
    
    f = interp1d(df.wavelength,df.flux, bounds_error=False)
    # note we are not propagating errors here
    return f(target_waves)

def build_pyspeckit(pars):
    for side in ['blue','red']:
        pars[side]['sp'] = []
        for i, _ in enumerate(pars[side]['files']):
            pars[side]['sp'].append(
                psk.Spectrum(xarr=pars[side]['raw_wavelength'][i],
                    data=pars[side]['raw_data'][i],
                    error=pars[side]['raw_error'][i],
                    header=pars['red']['header'][i],
                    xarrkwargs={'unit':'angstroms'},
                            unit='erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$'))


fit_pars = {
    6562: {'center':6562.8,
            'side':'red','xmin':6450,'xmax':6625,'exclude':[6520,6590]}
    }

def fit_three_gauss(sp,line=6562, i=0):

    fitfunc = 'threegauss'
    fname = make_fname(line=line, fitfunc=fitfunc, i=i)
    sp.plotter(xmin=fit_pars[line]['xmin'],xmax=fit_pars[line]['xmax'],
        errstyle='fill')
    sp.baseline(xmin=fit_pars[line]['xmin'], xmax=fit_pars[line]['xmax'],
            exclude=fit_pars[line]['exclude'],
            subtract=False, reset_selection=False,
            highlight_fitregion=True, order=2)
    T,F = True,False
    # this should be done automatically in pyspectkit fitters.py: it's a bug
    # TODO: read in the one Gauss component and use for guess/limits
    sp.specfit(fittype='gaussian', 
        guesses=[1e-16,line,3.5,5e-17,line,10,7e-17,line,9],
        limits=[(0,0), (line-25,line+25), (1.5,7), 
                (0,0), (line-25,line+25), (4,20),
                (0,0), (line-25,line+25), (4,20)],
        limited=[(T,F), (T,T), (T,T), 
                 (T,F), (T,T), (T,T),
                 (T,F), (T,T), (T,T)], renormalize=True, debug = True, verbose = True, use_lmfit = True)
    sp.specfit.plot_components(add_baseline=True)
    sp.plotter.savefig(fname+'.png')
    fitstr = sp.specfit.parinfo.__repr__()
    fitstr = fitstr[1:-1]
    with open(fname+'.par','w') as f:
        f.write(fitstr)
    plt.close()

def fit_two_gauss(sp,line=6562, i=0):

    fitfunc = 'twogauss'
    fname = make_fname(line=line, fitfunc=fitfunc, i=i)
    sp.plotter(xmin=fit_pars[line]['xmin'],xmax=fit_pars[line]['xmax'],
        errstyle='fill')
    sp.baseline(xmin=fit_pars[line]['xmin'], xmax=fit_pars[line]['xmax'],
            exclude=fit_pars[line]['exclude'],
            subtract=False, reset_selection=False,
            highlight_fitregion=True, order=2)
    T,F = True,False
    # this should be done automatically in pyspectkit fitters.py: it's a bug
    # TODO: read in the one Gauss component and use for guess/limits
    sp.specfit(fittype='gaussian', 
        guesses=[1e-16,line,3.5,5e-17,line,10],
        limits=[(0,0), (line-25,line+25), (1.5,7), 
                (0,0), (line-25,line+25), (4,20)],
        limited=[(T,F), (T,T), (T,T), 
                 (T,F), (T,T), (T,T)], renormalize=True, debug = True, verbose = True, use_lmfit = True)
    sp.specfit.plot_components(add_baseline=True)
    sp.plotter.savefig(fname+'.png')
    fitstr = sp.specfit.parinfo.__repr__()
    fitstr = fitstr[1:-1]
    with open(fname+'.par','w') as f:
        f.write(fitstr)
    plt.close()
    # instead, let's use sp.specfit.parinfo['SHIFT0'].value etc.
    # AMPLITUDE0, SHIFT0, WIDTH0...
#    fitpars = {}
#    vals = sp.specfit.parinfo.values
#    errs = sp.specfit.parinfo.errors
#    for i, par in enumerate(sp.specfit.parinfo.names):
#        fitpars[par] = {'value':vals[i], 'err':errs[i]}
#
#    return fitpars

def fit_gauss(sp,line=6562, i=0):

    fitfunc = 'gauss'
    fname = make_fname(line=line, fitfunc=fitfunc, i=i)
    sp.plotter(xmin=fit_pars[line]['xmin'],xmax=fit_pars[line]['xmax'],
        errstyle='fill')
    sp.baseline(xmin=fit_pars[line]['xmin'], xmax=fit_pars[line]['xmax'],
            exclude=fit_pars[line]['exclude'],
            subtract=False, reset_selection=False,
            highlight_fitregion=True, order=2)
    T,F = True,False
    sp.specfit(fittype='gaussian',
        guesses=[1e-16,line,3.5],
        limits=[(0,0), (line-25,line+25), (1.5,10)],
        limited=[(F,F), (T,T), (T,T)], debug = True, verbose = True) 
    sp.plotter.savefig(fname+'.png')
    fitstr = sp.specfit.parinfo.__repr__()
    fitstr = fitstr[1:-1]
    with open(fname+'.par','w') as f:
        f.write(fitstr)
    plt.close()
    # instead, let's use sp.specfit.parinfo['SHIFT0'].value etc.
#    fitpars = {}
#    vals = sp.specfit.parinfo.values
#    errs = sp.specfit.parinfo.errors
#    for i, par in enumerate(sp.specfit.parinfo.names):
#        fitpars[par] = {'value':vals[i], 'err':errs[i]}
#
#    return fitpars
        


def loop_fit(pars, fitfunc='gauss', line=6562):
    for i, sp in enumerate(pars[fit_pars[line]['side']]['sp']):
        assert(fitfunc in ['gauss', 'twogauss', 'threegauss'])
        if fitfunc == 'gauss':
            fit_gauss(sp,line=line, i = i)
        elif fitfunc == 'twogauss':
            fit_two_gauss(sp,line=line, i = i)
        elif fitfunc == 'threegauss':
            fit_three_gauss(sp,line=line, i = i)
    

def read_parfile(line=6562, fitfunc = 'gauss', i=0):
    fname = make_fname(line=line, fitfunc=fitfunc, i=i)
    with open(fname+'.par','r') as f:
        s = f.read()
        pars = s.split(', ')
        df = pd.DataFrame()
        for par in pars:
            tok = par.split()
            dfi = pd.DataFrame({'spectrum':int(i),
                'parname':tok[2],'value':float(tok[4]),'err':float(tok[6])},
                index=[0])
            df = df.append(dfi,ignore_index=True)

    return df

def read_fits(line=6562, fitfunc='gauss'):
    
    df = pd.DataFrame()
    # use the hard-coded values
    for i in range(47):
        dfi = read_parfile(line=line, fitfunc=fitfunc, i=i)
        df = df.append(dfi,ignore_index=True)

    return df



def make_fname(line=6562, fitfunc = 'gauss', i=0):
    return "fig/{:4d}/PTFS1623al_{:4d}_{}_{:03d}".format(line,line,fitfunc,i)

def get_fit_pars(pars, fitpar='SHIFT0', line=6562):
    raise ValueError('Deprecated: only works if fit was performed this session.  Use read_fits instead')
    vals = []
    errs = []
    for i, sp in enumerate(pars[fit_pars[line]['side']]['sp']):
        vals.append(sp.specfit.parinfo[fitpar].value)
        errs.append(sp.specfit.parinfo[fitpar].error)
    
    return np.array(vals), np.array(errs)
#Student, Ryan Jackim, edit: function to plot the data to make sure the data was being read in correctly. Delete at any time.
#def plot_parst(pars,df,fitpars=['WIDTH0']):

#    for fp in fitpars:
#        w = None
#        vals = None
#        errs = None
#        w = df['parname'] == fp
#        wp = filter(lambda x: x.startswith('SHIFT'),fitpars)
#        wp
#        vals = df.loc[w, 'value'].values
#        errs = df.loc[w, 'err'].values
#    plt.plot(vals,errs)
# rough TASC t0.  accretion disk is at compact object
t0 = 57695.256711278445
period=0.06633813731
def plot_pars(pars, df, fitpars=['SHIFT0','SHIFT1'], line=6562,
    period = period, t0 = t0, phased=False, fit_period=False):

    c = 2.99792458E5 #km/s

    side = fit_pars[line]['side']

    start = np.array(pars[side]['start_time']) 
    end = np.array(pars[side]['end_time']) 
    t = start + (end-start)/2.

    if fit_period:
        from gatspy.periodic import LombScargle
        for fp in fitpars:
            print(fp)
            w = df['parname'] == fp
            vals = df.loc[w, 'value'].values
            errs = df.loc[w, 'err'].values
            ls = LombScargle(Nterms=3).fit(t, vals, errs)
            ls.optimizer.period_range = (0.04, 0.085)
            period = ls.best_period

    if phased:
        assert period is not None
        assert t0 is not None
        x = ((t-t0) % period) / period
        xtit = 'Phase'
    else:
        t0 = t[0]
        x = (t-t0)*24. 
        xtit = 'Time (hours)'

    for fp in fitpars:
        w = df['parname'] == fp
        vals = df.loc[w, 'value'].values
        errs = df.loc[w, 'err'].values
        mask = np.where(errs < 10)
        print(vals[mask])
        print(errs[mask])
#        print(errs) what i need to do, is manipulate below to make each plot seperate for amp and shift
        if fp.startswith('SHIFT'):
            plt.errorbar(x, 
                (vals[mask]-fit_pars[line]['center'])/fit_pars[line]['center']*c,
                errs[mask]/fit_pars[line]['center']*c, fmt='.', label=fp, linestyle='none')
            plt.ylim(ymin = -700,ymax = 700)
            plt.xlabel(xtit)
            plt.ylabel('Velocity (km/s)')
        elif fp.startswith('AMPLITUDE'):
            plt.errorbar(x, vals, errs, fmt='.', label=fp, linestyle='none')
            plt.ylim(ymin = -2e-16,ymax = 2e-16)
            plt.xlabel(xtit)
            plt.ylabel('erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
        elif fp.startswith('WIDTH'):
            plt.errorbar(x, vals, errs, fmt='.', label=fp, linestyle='none')
            plt.ylim(ymin = -20,ymax = 20)
            plt.xlabel(xtit)
            plt.ylabel('Angstroms')

    sb.despine()
    
            

def stack_plot(pars,side='red', xrange = None, offset=1e-17):
    plt.figure()
    for i, start in enumerate(pars[side]['start_time']):
        x = pars[side]['raw_wavelength'][i]
        if xrange is None:
            xrange = [x.min(), x.max()]
        y = pars[side]['raw_data'][i]
        w = (x >= xrange[0]) & (x <= xrange[1])
        # figure out baseline, roughly
        if xrange is not None:
            ybg = np.percentile(y[w],5)
        else:
            ybg = 0
        plt.plot(x[w],y[w]-ybg+i*offset)




def plot_trailed(pars,side='red'):
    plt.figure()
    #norm = c.SymLogNorm(vmin=-0.25E-16,vmax=1.E-16,linthresh=1e-18)
    #norm = c.LogNorm(vmin=1E-18,vmax=1.E-16)
    norm = c.Normalize(vmin=-0.25E-16,vmax=1.E-16)

    T0 = pars[side]['start_time'][0]

    ymax = 0
    for i, start in enumerate(pars[side]['start_time']):
        x = pars[side]['raw_wavelength'][i]
        # meshgrid takes *bounds*,so we need to shift half a bin on both sides
        hwid = np.diff(x)/2.
        xl = x[1:] - hwid
        start = x[0] - hwid[0]
        end = x[-1] + hwid[-1]
        x2 = np.append(xl,end)
        x3 = np.insert(x2,0,start)
        y = [(pars[side]['start_time'][i] - T0)*24, 
            (pars[side]['end_time'][i] - T0)*24] 
        X, Y = np.meshgrid(x3, y)
        C = pars[side]['raw_data'][i]
        C = C.reshape(1,len(C))
        plt.pcolormesh(X,Y,C,edgecolors='None',cmap='Greys',
            norm=norm)
        ymax = y[1]
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Elapsed Time (hr)')
    if side =='red':
        xl = [6415, 8800]
    elif side == 'blue':
        xl = [3500, 5595]
    plt.xlim(xl)
    plt.ylim([0,ymax])

def plot_trailed_rebinned(pars):
    plt.figure()
    #norm = c.SymLogNorm(1).autoscale(pars['red']['median_data'])
    plt.imshow(pars['red']['median_data'],interpolation='none',cmap='Greys_r')
        #norm=norm)
    plt.figure()
    plt.imshow(pars['blue']['median_data'],interpolation='none',cmap='Greys_r')
        #norm=norm)

def plot_summed(pars):
    # plot with no correction for RVs
    plt.plot(pars['red']['wavelengths'],pars['red']['data'].sum(axis=0),color='black')
    plt.plot(pars['blue']['wavelengths'],pars['blue']['data'].sum(axis=0),color='black')

    sb.set_context("talk")#,font_scale=2)

    lines = (
    (r'H$\epsilon$',(3970.07)),
    (r'H$\delta$',(4101.76)),
    (r'H$\gamma$',(4340.47)),
    ('Bowen',(4640)),
    (r'H$\beta$',(4861.33)),
    ('He I',(3888.65)),
    ('He I',(4026)),
    ('He I',(4921)),
    ('He I',(5015)),
    ('He I',(5047)),
#    ('He I',(5875)),
    ('He I',(6678)),
    ('He I',(7065)),
    #('He I',(3888.65,4026, 4921, 5015, 5047, 5875.67,6678.152, 7065)),
    ('He II',(4686)),
    (r'H$\alpha$',(6562.8)),
    )

    ax = plt.gca()
    label_lines(lines, ax=ax, label_y = 5.5e-15,
            line_top = 5e-15, line_mid = 4.5e-15, line_bottom = 4e-15)
    plt.xlim([3000,9000])
    plt.ylim([1e-16,6e-15])
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    
    
    
