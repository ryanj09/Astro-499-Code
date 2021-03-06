
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
from io import StringIO
import requests

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
            
            sp = psk.Spectrum(xarr=pars[side]['raw_wavelength'][i],
                    data=pars[side]['raw_data'][i],
                    error=pars[side]['raw_error'][i],
                    header=pars['red']['header'][i],
                    xarrkwargs={'unit':'angstroms'},
                            unit='erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
            sp.specname = ''
            pars[side]['sp'].append(sp)

fit_pars = {
    6562: {'center':6562.8,
            'side':'red','xmin':6450,'xmax':6625,'exclude':[6520,6590]},
    6680: {'center':6678, 
            'side':'red','xmin':6615,'xmax':6750,'exclude':[6650,6700]},
    4861: {'center':4861.4,
            'side':'blue','xmin':4750,'xmax':4900,'exclude':[4825,4887]},
    4670: {'center':4686,
            'side':'blue','xmin':4550,'xmax':4800,'exclude':[4600,4750]},#(the exclude is where you make the fit defined)
    4340: {'center':4340.47,
            'side':'blue','xmin':4250,'xmax':4400,'exclude':[4300,4370]},
    4472: {'center':4472,
            'side':'blue','xmin':4420,'xmax':4540,'exclude':[4452,4500]},
    7065: {'center':7065,
            'side':'red','xmin':7000,'xmax':7150,'exclude':[7040,7090]},
    8345: {'center':8345,
            'side':'red','xmin':8335,'xmax':8355,'exclude':[8340,8350]},
    8400: {'center':8400,
            'side':'red','xmin':8360,'xmax':8420,'exclude':[8390,8410]}

    }

def get_fit_value(df, parname):
    return df.loc[df.parname == parname, "value"].values[0]

def get_fit_limits(df, parname):
    w = df.parname == parname
    rangelow = df.loc[w,'rangelow'].values[0]
    rangehigh = df.loc[w,'rangehigh'].values[0]
    islowlimit = np.isfinite(rangelow)
    ishighlimit = np.isfinite(rangehigh)
    if not islowlimit:
        rangelow = 0
    if not ishighlimit:
        rangehigh = 0
    return (rangelow,rangehigh),(islowlimit,ishighlimit)

def fit_three_gauss(sp,line=6562, i=0):

    fitfunc = 'threegauss'
    try:
        dfi = read_parfile(line=line, fitfunc = fitfunc, i=i)
        guesses = [get_fit_value(dfi, 'AMPLITUDE0'), get_fit_value(dfi, 'SHIFT0'), get_fit_value(dfi, 'WIDTH0'),
                get_fit_value(dfi, 'AMPLITUDE1'), get_fit_value(dfi, 'SHIFT1'), get_fit_value(dfi, 'WIDTH1'),
                get_fit_value(dfi, 'AMPLITUDE2'), get_fit_value(dfi, 'SHIFT2'), get_fit_value(dfi, 'WIDTH2')]
        limits = []
        limited = []
        for parname in ['AMPLITUDE0', 'SHIFT0', 'WIDTH0', 
                        'AMPLITUDE1', 'SHIFT1', 'WIDTH1',
                        'AMPLITUDE2', 'SHIFT2', 'WIDTH2']:
            limitsi, limitedi = get_fit_limits(dfi, parname)
            limits.append(limitsi)
            limited.append(limitedi)
    except FileNotFoundError:
        F = False
        T = True
        guesses=[1e-16,line,3.5,1e-16,line,3.5,5e-17,line,10]
        limits = [(0,0), (line-25,line+25), (1.5,7), 
                (0,0), (line-25,line+25), (1.5,7),
                (0,0), (line-25,line+25), (4,20)]
        limited = [(T,F), (T,T), (T,T), 
                 (T,F), (T,T), (T,T),
                 (T,F), (T,T), (T,T)]
    fname = make_fname(line=line, fitfunc=fitfunc, i=i)
    sp.plotter(xmin=fit_pars[line]['xmin'],xmax=fit_pars[line]['xmax'],
        errstyle='fill')
    sp.baseline(xmin=fit_pars[line]['xmin'], xmax=fit_pars[line]['xmax'],
            exclude=fit_pars[line]['exclude'],
            subtract=False, reset_selection=False,
            highlight_fitregion=True, order=1, annotate = False)
    T,F = True,False
    # this should be done automatically in pyspectkit fitters.py: it's a bug
    # TODO: read in the one Gauss component and use for guess/limits
    sp.specfit(fittype='gaussian', 
        guesses=guesses,
        limits=limits,
        limited=limited, renormalize=True, debug = True, verbose = True, use_lmfit = True, annotate = False)
    sp.specfit.plot_components(add_baseline=True)
    sp.plotter.savefig(fname+'.png')
    fitstr = sp.specfit.parinfo.__repr__()
    fitstr = fitstr[1:-1]
    with open(fname+'.par','w') as f:
        f.write(fitstr)
    plt.close()

def fit_two_gauss(sp,line=6562, i=0):

    fitfunc = 'twogauss'
    try:
        dfi = read_parfile(line=line, fitfunc = fitfunc, i=i)
        guesses = [get_fit_value(dfi, 'AMPLITUDE0'), get_fit_value(dfi, 'SHIFT0'), get_fit_value(dfi, 'WIDTH0'),
                get_fit_value(dfi, 'AMPLITUDE1'), get_fit_value(dfi, 'SHIFT1'), get_fit_value(dfi, 'WIDTH1')]
        limits = []
        limited = []
        for parname in ['AMPLITUDE0', 'SHIFT0', 'WIDTH0', 
                        'AMPLITUDE1', 'SHIFT1', 'WIDTH1']:
            limitsi, limitedi = get_fit_limits(dfi, parname)
            limits.append(limitsi)
            limited.append(limitedi)
    except FileNotFoundError:
        F = False
        T = True
        guesses = [1e-16,line,3.5,5e-17,line,10]
        limits = [(0,0), (line-25,line+25), (1.5,7), 
                (0,0), (line-25,line+25), (4,20)]
        limited = [(T,F), (T,T), (T,T), 
                 (T,F), (T,T), (T,T)]
    fname = make_fname(line=line, fitfunc=fitfunc, i=i)
    sp.plotter(xmin=fit_pars[line]['xmin'],xmax=fit_pars[line]['xmax'],
        errstyle='fill')
    sp.baseline(xmin=fit_pars[line]['xmin'], xmax=fit_pars[line]['xmax'],
            exclude=fit_pars[line]['exclude'],
            subtract=False, reset_selection=False,
            highlight_fitregion=True, order=1, annotate = False)
    T,F = True,False
    # this should be done automatically in pyspectkit fitters.py: it's a bug
    # TODO: read in the one Gauss component and use for guess/limits
    sp.specfit(fittype='gaussian', 
        guesses=guesses,
        limits=limits,
        limited=limited, renormalize=True, debug = True, verbose = True, use_lmfit = True, annotate = False)
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
    try:
        dfi = read_parfile(line=line, fitfunc = fitfunc, i=i)
        guesses = [get_fit_value(dfi,'AMPLITUDE0'), get_fit_value(dfi, 'SHIFT0'), get_fit_value(dfi, 'WIDTH0')]
        limits = []
        limited = []
        for parname in ['AMPLITUDE0', 'SHIFT0', 'WIDTH0']:
            limitsi, limitedi = get_fit_limits(dfi, parname)
            limits.append(limitsi)
            limited.append(limitedi)
    except FileNotFoundError:
        F,T = False,True
        guesses=[1e-16,line,3.5]
        limits = [(0,0), (line-25,line+25), (1.5,10)]
        limited = [(F,F), (T,T), (T,T)]
    fname = make_fname(line=line, fitfunc=fitfunc, i=i)
    sp.plotter(xmin=fit_pars[line]['xmin'],xmax=fit_pars[line]['xmax'],
        errstyle='fill')
    sp.baseline(xmin=fit_pars[line]['xmin'], xmax=fit_pars[line]['xmax'],
            exclude=fit_pars[line]['exclude'],
            subtract=False, reset_selection=False,
            highlight_fitregion=True, order=1, annotate = False)
    T,F = True,False
    sp.specfit(fittype='gaussian',
        guesses=guesses,
        limits=limits,
        limited=limited, debug = True, verbose = True, annotate = False) 
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
#    return fitpars twitch aueg

def fit_single(pars,i, fitfunc='gauss', line=6562):
    sp = pars[fit_pars[line]['side']]['sp'][i]
    assert(fitfunc in ['gauss', 'twogauss', 'threegauss'])
    if fitfunc == 'gauss':
        fit_gauss(sp,line=line, i = i)
    elif fitfunc == 'twogauss':
        fit_two_gauss(sp,line=line, i = i)
    elif fitfunc == 'threegauss':
        fit_three_gauss(sp,line=line, i = i)

def loop_fit(pars, fitfunc='gauss', line=6562):
    for i, sp in enumerate(pars[fit_pars[line]['side']]['sp']):
        assert(fitfunc in ['gauss', 'twogauss', 'threegauss'])
        if fitfunc == 'gauss':
            fit_gauss(sp,line=line, i = i)
        elif fitfunc == 'twogauss':
            fit_two_gauss(sp,line=line, i = i)
        elif fitfunc == 'threegauss':
            fit_three_gauss(sp,line=line, i = i)

def fit_display(line = 6562, fitfuncs = ['gauss','twogauss','threegauss']):
    f = open('fig/{:4d}/all_fits.html'.format(line), 'w')
    f.write('<html><BODY>\n')
    for i in range(47):
        f.write('<HR>{}<BR>\n'.format(i))
        for fitfunc in fitfuncs:
            image = "PTFS1623al_{:4d}_{}_{:03d}".format(line,fitfunc,i) + '.png'
            f.write("<IMG SRC = '{}' width = '400'>".format(image))
    f.write('</BODY></html>')
    f.close()
    

def read_parfile(line=6562, fitfunc = 'gauss', i=0):
    fname = make_fname(line=line, fitfunc=fitfunc, i=i)
    with open(fname+'.par','r') as f:
        s = f.read()
        pars = s.split(', ')
        df = pd.DataFrame()
        for par in pars:
            tok = par.split()
            tik = []
            for toki in tok:
                tik.extend([t for t in toki.split(':') if len(t)])
#            print(tik)
            pardic = {'spectrum':int(i),
                'parname':tok[2],'value':float(tok[4]),'err':float(tok[6])}
            if len(tik) == 9:
                rangestr = tik[8][1:-1]
                rangel = [float(r) for r in rangestr.split(',')]
                pardic["rangelow"] = rangel[0]
                pardic["rangehigh"] = rangel[1]
            dfi = pd.DataFrame(pardic,
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

def read_remapped_fits(line = 6562):
    r = requests.get('https://docs.google.com/spreadsheets/d/1WmS5gNuWP1G5nbIXfvfDmEIaVOZ2rLlboWwjchIpzMk/gviz/tq?tqx=out:csv&sheet='+str(line))
    df_map = pd.read_csv(StringIO(r.text))
    components = ['A','B','C']
    df = pd.DataFrame()
    for idx, row in df_map.iterrows():
        max_index = np.max([row[c] for c in components])
        if max_index == -1:
            continue
        elif max_index == 0:
            fitfunc = 'gauss'
        elif max_index == 1:
            fitfunc = 'twogauss'
        elif max_index == 2:
            fitfunc = 'threegauss'
        else:
            raise ValueError
        dfi = read_parfile(line=line, fitfunc=fitfunc, i=row['spectrum_number'])
        c2i = row[components].to_dict()
        i2c = {str(v):k for k, v in c2i.items()}
        for i,c in i2c.items():
            dfi['parname'] = dfi['parname'].str.replace(i,c, regex = False)
        df = df.append(dfi,ignore_index=True)
    return df

#now loop over rows in dfi and rename with i-c
#think about checking for duplicate assisgments


def make_fname(line=6562, fitfunc = 'gauss', i=0):
    return "fig/{:4d}/PTFS1623al_{:4d}_{}_{:03d}".format(line,line,fitfunc,i)

def make_fname_pars(line=6562, fitpars = ['SHIFT0'], phase = 0):
    return "fig/{:4d}/PTFS1623al_{:4d}_{}_phased_{}".format(line,line,fitpars,phase)

def make_fname_trailed(side = 'red', xl= [6415, 8800]):
    return "fig/PTFS1623al_trailed_{}_xlimit_{}".format(side,xl)

def get_fit_pars(pars, fitpar='SHIFT0', line=6562):
    raise ValueError('Deprecated: only works if fit was performed this session.  Use read_fits instead')
    vals = []
    errs = []
    for i, sp in enumerate(pars[fit_pars[line]['side']]['sp']):
        vals.append(sp.specfit.parinfo[fitpar].value)
        errs.append(sp.specfit.parinfo[fitpar].error)
    
    return np.array(vals), np.array(errs)

t0 = 57695.256711278445

period=0.06633813731
def plot_pars(pars, df, fitpars=['SHIFT0','SHIFT1'], line=6562,
    period = period, t0 = t0, phased=False, fit_period=False):

    c = 2.99792458E5 #km/s

    side = fit_pars[line]['side']
    start = np.array(pars[side]['start_time']) 
    end = np.array(pars[side]['end_time']) 

    t = start + (end-start)/2.
    sb.set_context("poster")#,font_scale=2)
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
        x_all = ((t-t0) % period) / period
        xtit = 'Phase'
        fname = make_fname_pars(line=line, fitpars='_'.join(fitpars), phase= period)
    else:
        t0 = t[0]
        x_all = (t-t0)*24. 
        xtit = 'Time (hours)'
        x_all = np.arange(len(x_all))
        fname = make_fname_pars(line=line, fitpars='_'.join(fitpars))
    for fp in fitpars:
        w = df['parname'] == fp
        x = x_all[df.loc[w,'spectrum']]
        vals = df.loc[w, 'value'].values
        errs = df.loc[w, 'err'].values
        mask = np.where(errs < 10)
        if fp.startswith('SHIFT'):
            print(max((vals[mask]-fit_pars[line]['center'])/fit_pars[line]['center']*c))
            print(min((vals[mask]-fit_pars[line]['center'])/fit_pars[line]['center']*c))
            plt.errorbar(x[mask], 
                (vals[mask]-fit_pars[line]['center'])/fit_pars[line]['center']*c,
                errs[mask]/fit_pars[line]['center']*c, fmt='.', label=fp, linestyle='none')
            plt.xlabel(xtit)
            plt.ylabel('Velocity (km/s)')
        elif fp.startswith('AMPLITUDE'):
            print(max(vals[mask]))
            plt.errorbar(x[mask], vals[mask], errs[mask], fmt='.', label=fp, linestyle='none')
            plt.xlabel(xtit)
            plt.ylabel('erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
        elif fp.startswith('WIDTH'):
            plt.errorbar(x[mask], vals[mask], errs[mask], fmt='.', label=fp, linestyle='none')
            plt.xlabel(xtit)
            plt.ylabel('Angstroms')
    sb.despine()
    plt.savefig(fname+ '.png', bbox_inches = 'tight')

    

def stack_plot(pars,side='red', xrange = None, offset=1e-17):
    plt.figure()
    for i, start in enumerate(pars[side]['start_time']):
        x = pars[side]['raw_wavelength'][i]
        if xrange is None:
            xrange = [x.min(), x.max()]
        y = pars[side]['raw_data'][i]
        w = (x >= xrange[0]) & (x <= xrange[1])
        if xrange is not None:
            ybg = np.percentile(y[w],5)
        else:
            ybg = 0
        plt.plot(x[w],y[w]-ybg+i*offset)




def plot_trailed(pars,side='red', xr = [6415, 8800], xb = [3500, 5595]):
    plt.figure()

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
        xl = xr #[6415, 8800]
    elif side == 'blue':
        xl = xb #[3500, 5595]
    fname = make_fname_trailed(side,xl)
    plt.xlim(xl)
    plt.ylim([0,ymax])
    plt.savefig(fname+ '.png', bbox_inches = 'tight')

def plot_trailed_rebinned(pars):
    plt.figure()
    #norm = c.SymLogNorm(1).autoscale(pars['red']['median_data'])
    plt.imshow(pars['red']['median_data'],interpolation='none',cmap='Greys_r')
        #norm=norm)
    plt.figure()
    plt.imshow(pars['blue']['median_data'],interpolation='none',cmap='Greys_r')
        #norm=norm)

def plot_summed(pars, xl = [3000,9000]):
    # plot with no correction for RVs
    plt.plot(pars['red']['wavelengths'],pars['red']['data'].sum(axis=0),color='black')
    plt.plot(pars['blue']['wavelengths'],pars['blue']['data'].sum(axis=0),color='black')

    sb.set_context("talk")#,font_scale=2)

    possible_lines = (
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
    lines = []
    for line in possible_lines:
        if (line[1] > xl[0]) and (line[1] < xl[1]):
            lines.append(line)

    fname = "fig/PTFS1623al_summed_xlimit_{}".format(xl)

    ax = plt.gca()
    label_lines(lines, ax=ax, label_y = 5.5e-15,
            line_top = 5e-15, line_mid = 4.5e-15, line_bottom = 4e-15)
    plt.xlim(xl)
    plt.ylim([1e-16,6e-15])
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    sb.despine()
    plt.savefig(fname+ '.png', bbox_inches = 'tight')
    
    
