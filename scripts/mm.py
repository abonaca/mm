from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy
from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord

import myutils

from os.path import expanduser
home = expanduser("~")

def fields_vdiff(stream='styx'):
    """"""
    t = Table.read('../data/{}_velocities_matt.fits'.format(stream))
    ind = t['vre']<20
    t = t[ind]
    
    rlim = {'ngc5466': 20.6, 'styx': 20}
    vbins = np.arange(-300,300,20)
    v = myutils.bincen(vbins)
    percentiles = [[16,84], [5,95], [0.5, 99.5]]
    
    fields = np.unique(t['field'])
    Nf = np.size(fields)
    
    poly_hb = np.loadtxt(home+'/observing/2017A_Hectochelle/data/{}_poly_hb.txt'.format(stream))
    poly_m = np.loadtxt(home+'/observing/2017A_Hectochelle/data/{}_poly_m.txt'.format(stream))
    
    path_hb = mpl.path.Path(poly_hb)
    path_m = mpl.path.Path(poly_m)
    
    dd = 2.5
    Ncol = 2
    plt.close()
    fig, ax = plt.subplots(Ncol, Nf, figsize=(Nf*dd, Ncol*dd), sharex='col')
    
    for i in range(Nf):
        infield = t['field']==fields[i]

        hd, be = np.histogram(t['vr'][infield], bins=vbins)
        
        bes = Table.read(home+'/observing/2017A_Hectochelle/data/besancon_{}_{:d}.txt'.format(stream, fields[i]), header_start=79, data_end=-4, format='ascii')
        bind = (bes['r']<rlim[stream])
        bes = bes[bind]
        points = np.array([bes['g-r'],bes['r']]).T
        cmdbox = path_m.contains_points(points) | path_hb.contains_points(points)
        
        Nobs = np.sum(infield)
        Nbes = np.sum(cmdbox)
        np.random.seed(98)
        Nrand = 200
        
        hb = np.zeros((Nrand, np.size(hd)))
        
        for j in range(Nrand):
            randind = np.random.randint(0, Nbes, Nobs)
            hb[j], be = np.histogram(bes['Vr'][cmdbox][randind], bins=vbins)
        
        plt.sca(ax[0][i])
        plt.plot(v, hd, 'k-', lw=2, label='Observed')
        plt.plot(v, np.median(hb, axis=0), 'r-', lw=2, zorder=0, label='Besanc$\c{}$on model')

        for p in percentiles:
            plt.fill_between(v, np.percentile(hb, p[0], axis=0), np.percentile(hb, p[1], axis=0), color='r', alpha=0.1)
        
        plt.title('Field {:.0f}'.format(fields[i]), fontsize='small')
        
        plt.sca(ax[1][i])
        plt.axhline(0, color='r', ls='-', lw=2, zorder=0)
        plt.plot(v, hd - np.median(hb, axis=0), 'k-', lw=2)
        for p in percentiles:
            plt.fill_between(v, hd - np.percentile(hb, p[0], axis=0), hd - np.percentile(hb, p[1], axis=0), color='k', alpha=0.1)
            
        plt.xlabel('$V_r$ (km/s)')
        
        if i==0:
            plt.sca(ax[0][i])
            plt.ylabel('Number')
            plt.legend(frameon=False, fontsize='x-small', handlelength=1)
            
            plt.sca(ax[1][i])
            plt.ylabel('$N_{obs}$ - $N_{Bes}$')
        else:
            plt.sca(ax[0][i])
            plt.setp(plt.gca().get_yticklabels(), visible=False)

            plt.sca(ax[1][i])
            plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig('../plots/{}_fields_vdiff.png'.format(stream))

    