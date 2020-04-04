
def trleg(ax, leg=[]):
    if len(leg)==0:
        ax.legend(bbox_to_anchor=[1.05,1], loc=2,frameon=False)        
    else:
        ax.legend(leg, bbox_to_anchor=[1.05,1], loc=2)

def hideax(ax):    
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax._frameon = False    

def hidespine(ax):    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
def axticks(ax,val, ntick):
    for v in val:
        ax.locator_params(axis=v,tight=True, nbins=5)
        
def saveit(f, fname):    
    f.savefig(fname, bbox_inches="tight")
