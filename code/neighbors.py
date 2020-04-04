import tables
import pandas as pd
import numpy as np
import pickle
import pdb
import csv
import datetime
import time
import os
import file_names
import ps
from scipy import sparse
def ctldat(ctlbins, binid,outcomes):
    ctldat = ctlbins.get_node("/" + binid)
    ctlids = ctldat[:,0]
    den = ctldat[:,1:9]
    ctldat = ctldat[:,9:]
    #ctldat = pd.DataFrame(trtinfo['scaler'].transform(ctldat),index=ctlids)

    omat = get_omat(ctlbins.get_node("/outcomes/" + binid), len(outcomes))

    return ctldat, ctlids, den,omat[:,np.array(['d'] + outcomes + ['a'])=="Lung_Cancer"].transpose()[0]
#ndir = "reproducibility/bupropion_example/"
import file_names
def get_sparse(ndir, drugid, bid, sparse_index):
    trt_h5 = tables.open_file(file_names.sparseh5_names(ndir, drugid),'r')
    trt_dense, trt_sparse = ps.load_selected(trt_h5, bid, sparse_index)
    from scipy import sparse
    trt_sparse = sparse.vstack(trt_sparse,format='csr')
    trt_dense = np.vstack(trt_dense)
    ix = np.argsort(trt_dense[:,0])
    trt_sparse = trt_sparse[ix,:]
    return trt_sparse, np.argsort(bid)

def get_omat(nodeinfo,omax):
    pdat = []

    for pato in list(nodeinfo):
        pato = list(pato)
        ixy = 1
        outs = [pato[0]]
        ogot = 0
        while ixy < len(pato):
            while pato[ixy] > ogot:
                outs += [0]
                #print("no {:d}, putting 0".format(ogot))                
                ogot += 1
            outs.append(pato[ixy + 1]) #if pato[ixy + 1] > 0 else 0
            #print(ogot, outs[-1], pato[ixy + 1], ixy)
            ogot += 1
            ixy += 2
        #outs = outs + [0]*(omax - ogot)
        pdat.append(outs + [0]*(omax - ogot))
    pdat = np.array(pdat) #pdat
    anyo = np.where(pdat[:,1:] > 0, pdat[:,1:], 5000).min(axis=1).reshape(-1,1)
    anyo = np.where(anyo==5000,0,anyo)
    return np.hstack((pdat,anyo))

def gen_embname(savename):
    return savename + ".binembeds.pytab"

def get_bin(ndir, binid, sparse_index,idin):
    
    trazbins = tables.open_file(gen_embname(ndir + "2293") ,mode="r")
    varenbins = tables.open_file(gen_embname(ndir + "2298") ,mode="r")
    bupbins = tables.open_file(gen_embname(ndir + "trt") ,mode="r")
    outcome = pickle.load(open(ndir + "outcomes_no_nonmel.pkl",'rb'))
    outcome = sorted(outcome.keys())
    #trt_compare, trtoutc = matching.binfo(trtinfo,binid )
    bpbin,bid,bden,blc = ctldat(bupbins,binid,outcome)
    trbin,tid,tden,tlc =ctldat(trazbins,binid,outcome)

    print("res:",len(tid), len(set(tid)))
    bidsparse, bix = get_sparse(ndir, 2305, bid, sparse_index)
    bpbin = bpbin[bix,:]
    bden = bden[bix,:]
    tidsparse, tix = get_sparse(ndir, 2293, tid, sparse_index)
    trbin = trbin[tix,:]
    
    tden = tden[tix,:]
    tid = tid[tix]
    bid = bid[bix]
    tlc = tlc[tix]
    blc = blc[bix]    
    '''
    '''
    print("res:",len(tid), len(set(tid)))
    print("stack res:",len(tid), len(set(tid) & set(bid)))
    #print("stack res:",len(tid), len(set(tid) & set(vid)))
    #print("stack res:",len(tid), len(set(bid) & set(vid)))

    vrbin,vid , vden,vlc = ctldat(varenbins,binid,outcome)
    vidsparse,vix = get_sparse(ndir, 2298, vid, sparse_index)
    vrbin = vrbin[vix,:]
    vden = vden[vix,:]
    vid = vid[vix]
    vlc = vlc[vix]
    
    X = np.vstack((bpbin, trbin,vrbin))
    den = np.vstack((bden,tden,vden))
    lab = np.hstack((np.zeros(bpbin.shape[0]),np.ones(trbin.shape[0]),np.ones(vrbin.shape[0]) + 1))
    sp = sparse.vstack((bidsparse, tidsparse, vidsparse),format='csr')

    ids = np.hstack((bid, tid, vid))
    lc = np.hstack((blc, tlc,vlc))
    #pdb.set_trace()
    '''

    X = np.vstack((bpbin, trbin))
    den = np.vstack((bden,tden))
    lab = np.hstack((np.zeros(bpbin.shape[0]),np.ones(trbin.shape[0])))
    sp = sparse.vstack((bidsparse, tidsparse),format='csr')
    print("lc",blc.shape, tlc.shape)
    lc = np.hstack((blc, tlc))
    ids = np.hstack((bid, tid))

    sel = np.isin(ids, idin)
    X = X[sel,:]; den = den[sel,:]; lab = lab[sel]; ids = ids[sel]; lc = lc[sel]; sp = sp[sel,:]
    '''
    print("res:",len(ids), len(set(ids)))
    return X, lab, sp, ids,den,lc

def sparsecol(si, code, sp):
    x = np.where(si['code']==code)[0]
    return np.array(sp[:,x].todense())


def get_matches1(ndir, ids,suff,i):
    nm = ndir + '2293.PSM' + str(i) + suff + '.ids.'
    t = np.loadtxt(nm +"trt" )
    c = np.loadtxt(nm +"ctl" )
    t = np.vstack((t,c)).transpose()

    t = t[np.isin(t[:,0],ids),:]
    return t


def embed_sparse(sp,sparse_index, femb, canc_index=np.zeros(0), weight=False):
    indices = sp.indices
    ptrs = sp.indptr
    eget = []
    if canc_index.shape[0] > 0:
        canc_index = canc_index.abs() #/canc_index.abs().sum()
    for i in range(sp.shape[0]):
        erows = sparse_index[indices[ptrs[i]:ptrs[i+1]]]
        times = sp.data[ptrs[i]:ptrs[i+1]]
        wt = np.exp(-1*(times**2)/50)
        if canc_index.shape[0] > 0:
            subdo = np.isin(erows, canc_index.index)
            #print("bef:",len(erows))
            erows = erows[subdo]
            times = times[subdo]
            wt = wt[subdo]
            #print("aft:",len(erows))            
            #wt = wt[subdo]*canc_index[erows]
            if weight:
                #pdb.set_trace()                
                #wt2 = np.multiply(wt,canc_index.loc[erows,'coef'].values)
                canc_coef = canc_index.loc[erows,'coef'].values
                for tu in set(times):

                    timeco = canc_coef[times==tu]
                    wt[times==tu] = np.multiply(wt[times==tu],timeco/timeco.max())
        #print(wt.shape, erows.shape)
        a=  (femb[erows.astype(int),:].transpose()*(wt + (10**-8)/(len(wt) if len(wt) > 0 else 1))/(wt.sum()+10**-8)).sum(axis=1)
        eget.append(a)
        #if i > 5:
        #    break
    return np.vstack(eget)

def neighborembs(res, embedding, ids):
    emb = pd.DataFrame(embedding, index = ids)
    y = list(zip(*tuple((np.ndarray.tolist(emb.loc[res[:,0],:].values), 
                         np.ndarray.tolist(emb.loc[res[:,1],:].values)))))
    return res, y

def embplt(embedding,elgrps,lcy=None):
    import plot_helper
    f, ax = plt.subplots(figsize=(10,10))
    ax.plot(embedding[:,0], embedding[:,1],'.k',markersize=2)
    cols = ['r','b','goldenrod']
    for i,c in enumerate(cols):
        ax.plot(embedding[lab==i,0],embedding[lab==i,1],'.',color=c,markersize=7 if i < 2 else 15,label=druglabs[i])
    shapes = "^sP"
    shapes = "x+"
    colors = ['c','violet']
    eli = 0
    for i,nm in elgrps.items():
        has = np.array(np.hstack((sp[:,np.where(si['code'].isin(nm))[0]] > 0).sum(axis=1)))[0,:]
        sel = np.where(has)[0]
        ax.plot(embedding[sel,0],embedding[sel,1],shapes[eli],color= colors[eli],markersize=10,
               label = i) #,markeredgecolor='y')
        didleg = True
        eli += 1 
    '''
    for i,nm in elgrps.items():
        has = np.array(np.hstack((sp[:,np.where(si['code'].isin(nm))[0]] > 0).sum(axis=1)))[0,:]
        didleg = False
        for idr, c in enumerate(cols):
            hasindrug = has & (lab==idr)
            print(druglabs[idr], i,hasindrug.sum(),embedding[np.where(hasindrug)[0],0].shape)
            if hasindrug.sum() > 0:
                sel = np.where(hasindrug)[0]
                ax.plot(embedding[sel,0],embedding[sel,1],shapes[eli],color= c,markersize=10,
                       label = '_nolegend_' if didleg else i,markeredgecolor='y')
                didleg = True
        eli += 1
    '''
    plot_helper.trleg(ax) #.legend(loc=2,bbox_to_center=[1,1.02])
    if lcy:
        lc = mc.LineCollection(lcy,colors = 'k',linewidths=1)
        ax.add_collection(lc)
    ax.set_xlim(np.percentile(embedding[:,0],[1,99]))
    ax.set_ylim(np.percentile(embedding[:,1],[1,99]))
    return f, ax
