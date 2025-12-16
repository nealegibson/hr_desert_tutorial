"""
Some basic inference functions adapted from my inferno module which should be available
here soon:
https://github.com/nealegibson/inferno
Really they are just rewritten versions of https://github.com/nealegibson/Infer
But there are many other options for optimisers/MCMCs/etc, and they should (in principle)
all do much the same job!

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from tqdm.auto import tqdm
import time

def fopt(f,x0,var=None,args=[],min=False,**kwargs):
  """
  Optimisation function using scipy's fmin.
  
  This uses a simple wrapper to allow maximising as well as minimising functions, as well
   as allowing for a fixed set of parameters. inferno.opt or inferno.optimise has more
   options.
  
  f - function to be optimised, which can be called as f(x0,*args)
  x0 - array of starting points
  var - array with the same length as x0 indicating variable parameters. Parameters are
    variable if >0, so boolean True/False, integers, or even error arrays will work
  args - additional arguments passed to fmin (see scipy.optimize.fmin)
  min - if True minimises rather than maximises the function f wrt x0
  kwargs - additional keyword args are passed to fmin
  
  """

  if var is None: var = np.ones(x0.size)
  var_ind = var>0
  x = np.copy(x0)
  
  #define wrapper function to re-construct the full parameter array from subset
  def wrapper(p,*args):
    x[var_ind] = p # get full parameter array
    if min: return f(x,*args) #call the function and return
    else: return -f(x,*args) #or the negative of the func to maximise the function instead
  
  #call scipy's fmin on the wrapper function
  x[var_ind] = fmin(wrapper,x0[var_ind],args=args,**kwargs)  
  return x 

def apply_uniform_logPrior(logL,bounds):
  """
  Simple funciton decorator that takes in a log Likelihood function, and returns a log Posterior
  that is bounded according to bounds.
  
  logL - input function of form logL(p,*args,**kwargs)
  bounds - list of (lower/upper) bounds for each parameter
  
  e.g. for a parameter vector of length 5, could use:
  bounds = [(0,1),(-np.inf,np.inf),None,(-10,10),(2,9)]
  
  then can define logP from logL as follows:
  logP = apply_uniform_prior(logL,bounds)
  
  logP is then a callable function just like logL
  logP(p,*args,...)
  
  """
  
  #convert bounds into lower and upper limits of len(p)
  lower_lims = np.array([i[0] if i is not None else -np.inf for i in bounds])
  upper_lims = np.array([i[1] if i is not None else np.inf for i in bounds])
  
  #define the posterior distribution
  def logP(p,*args,**kwargs):
    #apply priors - return -np.inf if outside prior range
    if np.any(p<lower_lims): return -np.inf
    if np.any(p>upper_lims): return -np.inf
    
    return logL(p,*args,**kwargs) #else return the logL
  
  return logP #return the function

def uniform_logPrior(bounds):
  """
  same as above, but returns the logPrior directly  
  """
  
  lower = np.array([t[0] if type(t) is tuple else -np.inf for t in bounds])
  upper = np.array([t[1] if type(t) is tuple else np.inf for t in bounds])
    
  def log_prior(x,*args,**kwargs):
    """
    logPrior with simple bounds
    """
    if np.any(x<lower): return -np.inf
    elif np.any(x>upper): return -np.inf
    else: return 0.
  
  return log_prior

def miniMCMC(f,x,x_err,burnin,chain,N=32,args=[],a=2,dlogP=50):
  """
  Simple MCMC function. This is missing a lot of functionality and checks of more
  extensive codes, see inferno.mcmc for a fully-featured mcmc implementation with multiple
  flavours of MCMC.
  
  This version runs a simple implementation of an Affine Invariant MCMC (this is a
  misnomer as most non-trivial Metropolis Hastings steps can be affine invariant!).
  However, this MCMC flavour does require minimal tuning, and therefore is good for
  testing. See Goodman & Weare (2010) for a description of the algorithm, and 
  Foreman-Mackey et al. (2012) for another clear description of the algorithm (and of the
  widely used 'emcee' implementation).
  
  The basic idea is that at each step, we loop through each chain in turn, and pick
  another random chain, and create a proposal based on the positions of the current and
  random other chain. This implementation splits the set of chains in two, and picks a
  random chain from the other set. This allows the proposals to be pre-computed, and also
  for the logPosteriors to be computed in parallel. See Foreman-Mackey et al. (2012) for
  an explanation.
  
  The initial set of chains are simply drawn from a diagonalised Gaussian distribution.
  Chains are replaced at random if more than dlogP from maximum likelihood to ensure the
  starting points are ok. If more than half the points are outside this range code will
  raise an exception. This is usually because the starting distribution is too large for
  one of the parameters, and/or the starting point is far from the max. This can usually
  be fixed by calling again with smaller x_err. Note that full mcmc implementation
  inferno.mcmc has many more features for refining starting points and running different
  flavours of MCMC.
  
  The total number of samples will be N * (burnin * chain)
  
  inputs
  ------
  
  f - logPosterior function, which can be called as f(x,*args), and returns the value
    of the logPosterior for the parameter vector x
  x - array of starting points (means) to populate initial chains
  x_err - corresponding uncertainties (stdevs)
  burnin - length of burnin period, where current state is not recorded
  chain - length of chain, where each state of the chain is recorded
  N - number of chains/walkers, must be even greater than 16
  args - additional arguments to the logPosterior f - i.e. f(x,*args)
  a - parameter used to control the acceptance ratio. This can be varied based on the
    acceptance, but is fixed here. See inferno.mcmc(mode="AffInv")
  dlogP - starting points for chains are rejected if more than dlogP from the maximum
    likelihood computed from the initial draw. This will include points in restricted
    prior space (ie with f(x)=-np.inf). If more than a quarter are outside this
    range, will raise an exception
  
  returns a dictionary with the following parameters
  --------------------------------------------------
  'p' - means of the parameters
  'e' - standard deviation of the parameters
  'chains' - array of chains of shape: (chains x N x len(x))
  'logP' - corresponding values of logP at each point in the chain, shape: (chains x N)
  'Xinit' - initial points in the chain, useful for debugging, shape: (N x len(x))
  'Xinit_logP' - corresponding values of logP, shape: (N)
  
  """

  #check a few inputs are ok
  assert N%2==0 and N>16, "N must be even and greater than 16"
  
  #define simple mapping function, written in this way to allow easy parallelisation with multiprocessing
  def f_args(x): return f(x,*args) #create simple wrapper function that doesn't require args
  def map_func(X): return np.array(list(map(f_args,X))) #create mapping function using default map
  
  #get starting points for the chains and compute logP for each
  X = np.random.multivariate_normal(x,np.diag(x_err**2),N) #use gaussian distribution
  XlogP = map_func(X) #compute logP
  Xinit,Xinit_logP=np.copy(X),np.copy(XlogP)
  
  #define arrays for chains
  chains = np.empty((chain,N,x.size)) # accepted chains
  logP = np.empty((chain,N)) # accepted posterior values
  n_steps = burnin+chain #and total number of steps
  acc = np.full((n_steps,N),False) # acceptance array, start with all False
  
  #re-draw starting points for outliers
  cull_index = XlogP.max() - XlogP > dlogP
  if np.sum(~cull_index) < np.sum(x_err>0.)*2: #raise exception if number of good points is too low
    raise ValueError("too many points ({}/{}) with ndim {} are outside acceptable range, use smaller x_err".format(np.sum(cull_index),len(cull_index),np.sum(x_err>0.)))
  if np.any(cull_index):
    print("redrawing {}/{} points".format(np.sum(cull_index),N))
    ind_good = np.where(~cull_index)[0]
    good_points_ind = np.random.choice(ind_good,cull_index.sum())
    X[cull_index],XlogP[cull_index] = X[good_points_ind],XlogP[good_points_ind]
  
  #predefine random arrays, for acceptance, step sizes, and random other chain
  RandNoArr = np.random.rand(n_steps,N) #for acceptance step
  #then z and z^D-1 used in proposal and acceptance
  z = (np.random.rand(n_steps,N) * (np.sqrt(4.*a)-np.sqrt(4./a)) + np.sqrt(4./a))**2 / 4.
  z_Dm1 = z**(np.sum(x_err>0.)-1)
  #pick random other chain to use for each step
  rand_chain = np.random.randint(0,N//2,(n_steps,N)) #first pick a random value from 0 to N//2 for each chain
  rand_chain[:,:N//2]+=N//2 #then add on N//2 for the 1st set
  slices = [slice(0,N//2),slice(N//2,None)]
  
  start_time = time.time() #get start time
  
  #compute MCMC chain
  for i in tqdm(range(n_steps),position=0,desc='running mcmc chain'):
    for sl in slices: #loop over each half of chains in turn
      #get proposal steps and compute logP
      X_prop = X[rand_chain[i,sl]] + z[i][sl,np.newaxis] * (X[sl] - X[rand_chain[i,sl]])
      XlogP_prop = map_func(X_prop) #compute logP for proposal steps
      #accept or reject proposal steps
      accepted = RandNoArr[i,sl] < z_Dm1[i,sl] * np.exp(XlogP_prop - XlogP[sl])
      X[sl][accepted] = X_prop[accepted]
      XlogP[sl][accepted] = XlogP_prop[accepted]
      #store results in chain/acceptance arrays
      if i >= burnin:
        acc[i-burnin,sl] = accepted
        chains[i-burnin,sl] = X[sl]
        logP[i-burnin,sl] = XlogP[sl]

  ts = time.time() - start_time
  print('Total time: {:.0f}m {:.2f}s'.format(ts // 60., ts % 60.))      
  print("Final acceptance = {}%".format(acc.sum()/acc.size*100))
      
  return dict(p=chains.mean(axis=(0,1)),e=chains.std(axis=(0,1)),chains=chains,logP=logP,Xinit=Xinit,Xinit_logP=Xinit_logP)

def miniSamplePlot(X,N=None,labels=None,samples=300,x=None,left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.03,hspace=0.03):
  """
  Create a simple plot of MCMC chains
  
  X - chains (chain_length x N_chains x N_pars)
  N - reshape into N pseudo chains
  labels - labels for each parameter
  samples - no of samples to plot from each chain
  
  """
  
  assert X.ndim==3
  if N is not None and N>1:
    X = X.reshape(-1,N,X.shape[-1])
  else: N = X.shape[1]
        
  #define labels if None
  if labels is None: labels = [r'$\theta_{{{}}}$'.format(i) for i in range(X.shape[-1])]
  
  #create filter for any fixed parameters
  filt = ~np.isclose(np.std(X,axis=(0,1)),0)
  S = X[...,filt]
  labels = np.array(labels)[filt]
  if x is not None: x = x[filt]
  
  #first get the axes
  plt.figure()
  ax = {}
  
  D = filt.sum() #number of variable parameters
  for i in range(D): #loop over the parameter indexes supplied
    for q in range(i+1):
      ax['{}{}'.format(i,q)] = plt.subplot(D,D,i*D+q+1,xticks=[],yticks=[])
      if i == (D-1): ax['{}{}'.format(i,q)].set_xlabel(labels[q])
    ax['{}{}'.format(i,0)].set_ylabel(labels[i])
 
  #do histograms
  for n in range(N): #loop over chains
    for i in range(D): #loop over parameters
      ax['{}{}'.format(i,i)].hist(S[:,n,i],20,histtype='step',density=1)
      if n==0 and x is not None: ax['{}{}'.format(i,i)].axvline(x[i],color='0.5',lw=1,ls='--')
  
  #do scatter plots
  for n in range(N): # loop over chains
    ind = np.random.randint(0,X.shape[0],samples)
    #loop over the axes (except-diagonals) and make scatter plot
    for i in range(D): #loop over the parameter indexes supplied
      for q in range(i):
        ax['{}{}'.format(i,q)].plot(S[:,n,q][ind],S[:,n,i][ind],'o',ms=3,alpha=0.3)
        if n==0 and x is not None:
          ax['{}{}'.format(i,q)].axvline(x[q],color='0.5',lw=1,ls='--')
          ax['{}{}'.format(i,q)].axhline(x[i],color='0.5',lw=1,ls='--')
        if n==N-1:
          ax['{}{}'.format(i,q)].set_xlim(ax['{}{}'.format(q,q)].set_xlim())
          
  
  plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)

