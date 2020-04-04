import matplotlib.cm as cmx
import numpy as np
import pandas as pd
from itertools import chain
import pdb
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from matplotlib.colors import LogNorm
import matplotlib.colors as colors


def clust2way(class_cor,colormap_string, cmax = 1, cmin=None, cluster_columns = True, no_row_dendrogram= False ,no_col_dendrogram= False , nanval=0,hide_top_axis=True,tickfont=8,sym=False,sz=5,dotslab=np.zeros(0)): #cmap = 'bone_r', normer=LogNorm(vmin=10**-5, vmax=.1)):
    naned = class_cor.copy()
    naned[pd.isnull(naned)]= nanval
    imgheight = .6
    imgwidth = .6
    imgbottom = .1    
    imgleft = .09        
    fig = plt.figure(figsize=(sz*3,sz*2),dpi=300)
    
    Y = sch.linkage(naned, method='complete') #,metric='correlation')
    ## add_axes([l, b, w, h])
    
    ax = fig.add_axes([0.7,imgbottom,0.08,imgheight])
    Z1 = sch.dendrogram(Y, orientation='right',no_plot=no_row_dendrogram)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    yidx = Z1['leaves']
    #idx1 = idx1[-1:0:-1]
    yidx.reverse()

        
    xidx = range(class_cor.shape[1])
    if not sym and cluster_columns:
        ax = fig.add_axes([imgleft,imgbottom + imgheight,imgwidth,.2])
        horztree = sch.linkage(naned.transpose(), method='complete')
                               #metric = 'correlation') #lambda u,v: stats.spearmanr(u,v)[0])    
        Z2 = sch.dendrogram(horztree, no_plot=no_col_dendrogram) #orientation='top')
        if hide_top_axis:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()

        xidx = Z2['leaves']
        #idx1 = idx1[-1:0:-1]
        #xidx.reverse()
    
    axmatrix = fig.add_axes([0.09,imgbottom,imgwidth,imgheight])
    '''
    norm = colors.Normalize(vmin=-.3, vmax=.3)
    cmap = cmx.bwr
    m = cmx.ScalarMappable(norm=norm, cmap=cmap)
    '''
    if cmin is None: cmin  = -1*cmax
    #cmap.set_bad('white',1.)
    if sym:
        xidx = yidx
    a = class_cor.iloc[yidx, xidx]
    #return a
    masked_array = np.ma.array (a, mask=pd.isnull(a))
    cmap = cmx.get_cmap(colormap_string)
    print( pd.isnull(a).sum().sum())
    cmap.set_bad('black',1.)

    cax = axmatrix.imshow(masked_array,  cmap=cmap,vmin=cmin, vmax = cmax, #-1*absmax, vmax=absmax,
                          interpolation='nearest', aspect = 'auto')
    #cax = axmatrix.imshow(class_cor.iloc[yidx, xidx], #yidx
    #                      cmap=colormap_string,vmin=cmin, vmax = cmax, #-1*absmax, vmax=absmax,
    #                      interpolation='none', aspect = 'auto')
    
    axmatrix.set_xticks(range(class_cor.shape[1]))
    axmatrix.set_yticks(range(class_cor.shape[0]))
    axmatrix.set_xticklabels(class_cor.iloc[:,xidx].columns,rotation=90,fontsize=tickfont)
    axmatrix.set_yticklabels(class_cor.iloc[yidx,:].index,fontsize=tickfont) #.iloc[yidx,:]
    ylm = axmatrix.get_ylim()
    #axmatrix.plot([0,10],[0,20],c='k',linewidth=5)
    if dotslab.shape[0] > 0:
        print("adding dots",len(dotslab),len(set(dotslab)))
        dotslab = np.array(dotslab[yidx])
        print(dotslab)
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=len(set(dotslab)))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        axmatrix.scatter(np.ones(dotslab.shape[0]), np.arange(class_cor.shape[0]),
                         c = dotslab,
                         vmin=0, vmax=len(set(dotslab)),cmap=jet)
        axmatrix.set_ylim(ylm)
        #for i in range(len(dotslab)):
        #    #print(i, scalarMap.to_rgba(dotslab[i]))
        #    axmatrix.plot(1,i,color=scalarMap.to_rgba(dotslab[i]),markersize=5)


    axcbar = fig.add_axes([0.73 if not no_row_dendrogram else .65 ,0.05,0.1,0.6])
    axcbar.set_xticks([])
    axcbar.set_yticks([])
    #axcbar.set_axes([])
    axcbar.set_axis_off()
    cbar = fig.colorbar(cax, ax=axcbar,ticks=[round(i,2) for i in np.linspace(cmin,cmax,7)]) #rientation='horizontal',

    #tk = cbar.get_ticks()
    #cbar.set_ticklabels(10**

    return fig, axmatrix, xidx, yidx


def img_only(axmatrix, fig, class_cor, colormap_string, cmin, cmax,
                xticks=True):
    cax = axmatrix.imshow(class_cor, #yidx
                          cmap=colormap_string,vmin=cmin, vmax = cmax, #-1*absmax, vmax=absmax,
                          interpolation='none',
                            aspect = 'auto') #equal',extent=[0, class_cor.shape[1]-1,0, class_cor.shape[1]-1]) #auto
    if xticks:
        axmatrix.set_xticks(range(class_cor.shape[1]))
        axmatrix.set_xticklabels(class_cor.columns,rotation=90)
    else:
        axmatrix.set_xticks([])
        
    axmatrix.set_yticks(range(class_cor.shape[0]))    
    axmatrix.set_yticklabels(class_cor.index) #.iloc[yidx,:]
    

    axcbar = fig.add_axes([.85, #0.95 if not no_row_dendrogram else .85 ,
                            0.05 if xticks else .2,0.1,0.6])
    axcbar.set_xticks([])
    axcbar.set_yticks([])
    #axcbar.set_axes([])
    axcbar.set_axis_off()
    cbar = fig.colorbar(cax, ax=axcbar,ticks=[round(i,2) for i in np.linspace(cmin,cmax,7)]) #rientation='horizontal',

    
