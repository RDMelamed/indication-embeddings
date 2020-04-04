from statsmodels.base.model import GenericLikelihoodModel
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import nbinom

import numpy as np
from scipy.stats import nbinom
def ll_nbd(obs, expe, b1, a1):
    ll = nbinom.logpmf(obs, a1, 1 - expe/(b1+expe))
    #ll = nbinom.logpmf(y, size, prob)
    return ll


class NBin(GenericLikelihoodModel):
     def __init__(self, endog, exog, **kwds):
         super(NBin, self).__init__(endog, exog, **kwds)
     def nloglikeobs(self, params):
         alph = params[0]
         beta = params[1]
         #print str(alph) + ' ' + str(beta)
         ll = ll_nbd(self.endog, self.exog.transpose(), beta, alph).transpose()
         #print ll[:5]
         return -ll
     def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
         if start_params == None:
             # Reasonable starting values
             start_params = np.append(np.zeros(self.exog.shape[1]), .5)
             start_params[0] = np.log(self.endog.mean())
         return super(NBin, self).fit(start_params=start_params,
                                      maxiter=maxiter, maxfun=maxfun,
                                      **kwds)

def do_gps(obsexp, p):
    forgamma = pd.DataFrame({'shape':obsexp['observed'] + p.params[0],
                             'scale':1/(obsexp['expected'] + p.params[1])})
    resgamma = forgamma.apply(lambda x:stats.gamma(x[1],loc=0,
                                                    scale=x[0]).ppf([.05,.95]),axis=1).rename(columns={'scale':'ebg.05','shape':'ebg.95'})
    return pd.concat((obsexp, forgamma, resgamma),axis=1)

