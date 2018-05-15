"""plotting module
"""

# import sys
# import os
# import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
# import imp

def data_plot(df,x,y,color='b',label='_nolegend_',x_bins=None,ax=None):

    df_sub = df[[x,y]]

    if x_bins is None:
        x_bins = np.sort(df_sub[x].astype(int).unique()).astype(float)
    bin_width = x_bins[1] - x_bins[0]
    x_edge = np.arange(x_bins[0]-bin_width/2.0,x_bins[-1]+bin_width/2.0 + bin_width,bin_width)

    if not ax:
        fig, ax = plt.subplots()


    # Add and remove the binned column
    df_sub['bin_val'] = pd.cut(df_sub[x], bins=x_edge,labels=x_bins,right=True)

    # Get some stats
    df_stat = df_sub[[y,'bin_val']].groupby(['bin_val']).agg([np.mean,np.std,lambda x: scipy.stats.sem(x, ddof=1) * 1.96])#   [[.groupby()
    df_stat.columns = ['mean_val','std_val','ci']# df_stat.columns.droplevel()
    df_stat = df_stat.reset_index()
    df_stat['bin_val'] = df_stat.bin_val.astype(float)

    # Plot the underlying points
    ax.scatter(df_sub[x],df_sub[y],color=color,label='_nolegend_',alpha=0.01,s=0.5,marker='.')

    # Plot the main trend
    # print(df_stat.bin_val)
    ax.plot(df_stat.bin_val,df_stat.mean_val,label=label,color=color)
    ax.fill_between(df_stat.bin_val,df_stat.mean_val-df_stat.std_val,df_stat.mean_val+df_stat.std_val,alpha=0.2,color=color,label='_nolegend_')

    # print(df_stat.head())


    



def pat_plot(time,data,category,name_first_dir='',name_second_dir='',name_category='',window=60,
             threshold=.1,confidence=99,size=10,alpha=.1,size_ma=50,alpha_ma=.3,sizeratio=5,
             ybounds=None,matrue=True,horiz180true=False,lognormcolortrue=False,
             labelstrue=True,turnofftruedata=True,changesizema=True,
             title=None,ylabel=None,colormap='viridis'):
    #20180504 11:58am
    import matplotlib
    import matplotlib.colors as mcolors
    from numpy import inf
    data[data == inf] = np.nan
    data[data == -inf] = np.nan
    
    if window is None:
        window = 60
        
    mirroryboundstrue = False
    if ybounds is not None:
        mirroryboundstrue = True

    def pat_confidence_interval(sample_mean,sample_std,number_samples,confidence,twotailed):
        #20180427 4:23pm
        confidence_percent = confidence/100.
        alpha = 1.-confidence_percent
        if twotailed == True:
            alpha = alpha/2.
        tc = scipy.stats.t.ppf(1-alpha, number_samples)
        low_bound = sample_mean-(tc*(sample_std/np.sqrt(number_samples)))
        up_bound = sample_mean+(tc*(sample_std/np.sqrt(number_samples)))
        error = up_bound-low_bound
        return low_bound,up_bound,error  

    def make_colormap(seq):
        #borrowed heavily from https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        """Return a LinearSegmentedColormap
        seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        """
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)
    c = mcolors.ColorConverter().to_rgb
    patcolormap = make_colormap([c('black'),
                                 .1,
                                 c('black'),
                                 
                                 c('red'),
                                 .3,
                                 c('red'),
                                 
                                 c('orangered'),
                                 .35,
                                 c('orangered'),
                                 
                                 c('yellow'),
                                 .5,
                                 c('yellow'),
                                 
                                 c('limegreen'),
                                 .65,
                                 c('limegreen'),
                                 
                                 c('cornflowerblue'), 
                                 .7,
                                 c('cornflowerblue'), 
                                 
                                 c('black'),
                                 .9,
                                 c('black')])
    
    def moving_average(time,timeseries,window):
        print('window',window)
        half_window = window//2
        index = half_window
        means = []
        sizes_not_nan = []
        for i in range(len(timeseries)):
            means.append(np.NAN)
            sizes_not_nan.append(np.NAN)
        for i in time[half_window:len(time)-half_window]:
            number_not_nan = np.count_nonzero(~np.isnan(timeseries[index-half_window:index+half_window]))
            if number_not_nan/window > threshold:
                means[index] = np.nanmean(timeseries[index-half_window:index+half_window])
                sizes_not_nan[index] = sizeratio*number_not_nan
            index = index + 1
        return means,sizes_not_nan
    
    def moving_std(time,timeseries,window):
        half_window = window//2
        index = half_window
        means = []
        stds = []
        for i in range(len(timeseries)):
            means.append(np.NAN)
            stds.append(np.NAN)
        for i in time[half_window:len(time)-half_window]:
            number_not_nan = np.count_nonzero(~np.isnan(timeseries[index-half_window:index+half_window]))
            if number_not_nan/window > threshold:
                means[index] = np.nanmean(timeseries[index-half_window:index+half_window])
            index = index + 1
        half_window = window//2
        index = half_window
        for i in time[half_window:len(time)-half_window]:
            number_not_nan = np.count_nonzero(~np.isnan(timeseries[index-half_window:index+half_window]))
            if number_not_nan/window > threshold:
                stds[index] = np.sqrt(np.nansum((timeseries[index-half_window:index+half_window]-means[index-half_window:index+half_window])**2)/(float(number_not_nan)-1))
            index = index + 1
        return stds
    
    #######ACTUAL PLOTTING STUFF HERE
    fig, ax = plt.subplots(figsize = (17,8))
    #patnorm is necessary for the lognormcolor part
    patnorm = matplotlib.colors.NoNorm()
    if lognormcolortrue == True:
        colormap = patcolormap
        if turnofftruedata != True:
            plt.scatter(time, np.array(data), s = size/5, marker = 'x', c = 'r', alpha = alpha, zorder = 9)
        plt.scatter(time, np.array(data), s = size, marker = '.', c = category, alpha = alpha, cmap = colormap, norm = matplotlib.colors.SymLogNorm(linthresh = 10),zorder = 10)
        cbar = plt.colorbar()  # show color scale
        cbar.set_label(name_category)
        cbar.solids.set(alpha = 1)
    elif lognormcolortrue == False:
        if turnofftruedata != True:
            plt.scatter(time, np.array(data), s = size/5, marker = 'x', c = 'r', alpha = alpha, zorder = 9)
        plt.scatter(time, np.array(data), s = size, marker = '.', c = category, alpha = alpha, cmap = colormap, zorder = 10)
        cbar = plt.colorbar()  # show color scale
        cbar.set_label(name_category)
        cbar.solids.set(alpha = 1)
    #moving average
    if matrue == True:
        ma,sizes_not_nan = moving_average(time,data,window)
        std = moving_std(time,data,window)
        low_bound, up_bound, error = pat_confidence_interval(ma, std, window, confidence, True)
        if changesizema == True:
            plt.scatter(time, ma, s = sizes_not_nan, c = category, alpha = alpha_ma, cmap = colormap, norm = matplotlib.colors.SymLogNorm(linthresh = 10),zorder = 8)
            #make legend label
            plt.scatter('2018-01-01 23:50:00',0,label = 'Size of Moving Average using' + '\n' + 'Full Data in Window', s = np.nanmax(sizes_not_nan),color = 'gray')
        if changesizema != True:
            plt.scatter(time, ma, s = size_ma, c = category, alpha = alpha_ma, cmap = colormap, norm = matplotlib.colors.SymLogNorm(linthresh = 10),zorder = 8)
        plt.errorbar(time, ma, yerr = error, linestyle = '', color = 'black', alpha = .1)

    else:
        plt.ylim(-abs(np.nanmax(np.array(data)))-5,abs(np.nanmax(np.array(data)))+5)
    #horizontal lines at 180 and -180
    if horiz180true == True:
        ax.axhline(180, xmin = 0, xmax = 1, color = 'black', linestyle = '--', alpha = .8)
        ax.axhline(-180, xmin = 0, xmax = 1, color = 'black', linestyle = '--', alpha = .8)
    #average
    ax.axhline(0, xmin = 0, xmax = 1, color = 'black', linestyle = '-', alpha = .4)
    ax.axhline(np.nanmean(np.array(data)), xmin = 0, xmax = 1, color = 'red', linestyle = '--', alpha = .8)
    if labelstrue == True:
        plt.title('Relative Position' + '\n' + '(+ = degrees ' + name_first_dir + ' is to left of ' + name_second_dir + ')' + '\n' + '(- = degrees ' + name_first_dir + ' is to right of ' + name_second_dir + ')')
        plt.ylabel('Degrees Difference')
    else:
        plt.title(title)
        plt.ylabel(ylabel)
    plt.grid(which = 'both', linestyle = '--', alpha = .6)
    plt.autoscale()
    plt.xlim(np.min(time),np.max(time))
    if mirroryboundstrue == True:
        plt.ylim(-ybounds,ybounds)
    if (mirroryboundstrue != True) and (matrue == True):
        ma[ma == inf] = np.nan
        ma[ma == -inf] = np.nan
        error[error == inf] = np.nan
        error[error == -inf] = np.nan
        maxerrorbar = np.nanmax(np.abs(np.array(ma))+np.array(error))
        plt.ylim(-maxerrorbar-.1*maxerrorbar,maxerrorbar+.1*maxerrorbar)
    plt.legend(loc = 'best')
    plt.show()
    
    if labelstrue == True:
        print(name_first_dir + ' is to left of ' + name_second_dir + ' by (on average during timeframe):',np.round(np.nanmean(data),3),'degrees (red dotted line)')