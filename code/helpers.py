import pandas as pd
import sys
import numpy as np
import pickle
import os
from collections import defaultdict
## parse medi file
def medi_map(medi,icd2phe):
    noprob = []
    icd = []

    def isfloat(value):
      try:
        float(value)
        return True
      except ValueError:
        return False

    for i in medi['ICD9']:
        if not '-' in i:
            noprob.append(True)
            icd.append([j for j in icd2phe if j.startswith(i)])
        else:
            rlo, rhi = i.split("-")
            try:
                rlo = float(rlo)
                rhi = float(rhi)
                noprob.append(True)
                icd.append([j for j in icd2phe if isfloat(j) and float(j) >= rlo and float(j) <= rhi])
            except ValueError: 
                if rlo[0] == rhi[0]:
                    noprob.append(True)
                    lstart = rlo[0][0]
                    rlo = float(rlo[1:])
                    rhi = float(rhi[1:])
                    icd.append([j for j in icd2phe if j[0]==lstart and float(j[1:]) >= rlo and float(j[1:]) <= rhi])
                else:
                    noprob.append(False)
    medi['icd-parsed'] = noprob

    (g2filtingname, g2filtingcui) = pickle.load(open('annotations/gennme2ingred.pkl','rb'))
    from collections import defaultdict
    ingcui2gen = defaultdict(set)
    for g, c in g2filtingcui.items():
        for cdo in c:
            ingcui2gen[cdo] |= set([g])    
    

    medi_all = defaultdict(set)
    for i,g in enumerate(medi.index): #enumerate(medi['gennme']):
        for gennme in ingcui2gen[str(medi.loc[g,'RXCUI_IN'])]:
            medi_all[gennme] |= set(icd[i])
    return medi_all

def parse_medi(medifile, filter_many =True):
    ## mapping of all ICD codes to a phenotype
    icd2phe = pickle.load(open("annotations/icd2phe.03.18.pkl",'rb'))
    
    medi_hps = medi_map(pd.read_table(medifile,sep=","),icd2phe)

    hps_icd2g = defaultdict(set)
    for k,v in medi_hps.items():
        ### remove drugs that are prescrbed for more than 2% or so of all different *diseases* 
        ### (not removing drugs with )
        if filter_many and len(set([icd2phe[ic] for ic in v])) > 15:
            print('skipping', k)
            continue
        for ic in v:
            hps_icd2g[ic].add(k)
    fname = medifile.replace(".csv","") + ("" if filter_many else "_nofilt") +".pkl"
    f=  open(fname,'wb')
    pickle.dump(hps_icd2g, f)
    f.close()
    return fname

