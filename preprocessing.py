
import numpy as np

def pca(X,N=1,mean_sub=True,norm_cols=False,add_bias=False,ret_eigenvals=False):

  """
  Simple version of PCA to extract N orthonormal basis vectors U for a dataset X, as well
    well as reconstruct the dataset using those vectors.
    
  In summary performs SVD, which decomposes X into U D V.T
  Or using np version: u,d,vt = np.linalg.svd(D) # note vt is the tranpose of standard
  Then can reconstruct X using u @ diag(d) @ vt, or a lower rank version via
    u[:,:N] @ diag(d[:N]) @ vt[:N]
  
  X can also be a 3D array, in which case PCA is performed independently over each X[i]
  
  Note that U are orthonormal. The bias term is added after SVD, but as long as the mean
    subtraction is already done, the remaining vectors should be orthogonal. The bias term
    is rescaled to be normalised. You can check using U.T @ U which should equal I
    
  As the U and W vectors (before rescaling for returning) are orthonormal, the variance
    of each outer product of u w.T = 1. This means that for each component
    lambda_i * u w.T, lambda_i / np.sqrt(NL) = RMS of subtracted (image) component.
    
  To recover the eigenvalues of the SVD of X, can use l = np.sqrt(np.diag(W.T @ W)). Then
    the RMS of each map component is equal to l / np.sqrt(NL). Alternatively can provide
    ret_eigenvals=True and eigenvalues are returned as the final argument.
  
  (where NL = size of each 'image' that is decomposed, ie NL = X.size if X is 2D)
  
  inputs
  ------
    
  X - input data. Can be 2D or 3D. If the latter, pca is performed over each X[i]
  N - number of components to use in the fit/reconstruction
  mean_sub - subtract the mean before applying PCA.
  norm_cols - divide by stdev before applying SVD. This is equivalent to using the stdev
    of each columns as the uncertainties in the 'fit'
  add_bias - this adds a column of ones to each set of basis vectors U. This is useful if
    re-fitting the basis vectors to the data or another model  
  
  outputs
  -------
  
  U - set of basis vectors [for each X[i] if 3D]. A column of constants is appended if add_bias=1
    [rescaled so that vector is normalised]
  W - basis vectors of shape [(order x) wavelength x N] or [(order x) wavelength x N+1] if add_bias
    [these are rescaled by the eigenvectors so maps can be easily constructed from them]
  M - reconstructed model [for each X[i] if 3D]
  l[optional] - eigenvalues if ret_eigenvals is True. In this case W is not rescaled.
    
  """
  
  #subtract mean
  if mean_sub:
    Xmean = np.mean(X,axis=-2) #get mean of dataset along final axis
  else: Xmean = np.zeros([X.shape[0],X.shape[-1]] if X.ndim==3 else [X.shape[-1],])
  R = X - Xmean[...,None,:]

  if norm_cols: # divide through by stdev of each column to account for uncertainties
     stdev = np.std(X,axis=-2) # get stdev of dataset
     with np.errstate(divide='ignore',invalid='ignore'): R /= stdev[...,None,:]
     R[np.isnan(R)] = 0. # in case of division by zero
  
  #run SVD to get basis vectors - can do 3D arrays at once [1st axis independent]
  U,d,Vt = np.linalg.svd(R)
  
  #reconstruct model with N basis vectors. could do in single line with einsum
  if X.ndim==2:
    M = np.dot(U[:,:N]*d[:N],Vt[:N])
  elif X.ndim==3:
    M = np.array([np.dot(U[i,:,:N]*d[i,:N],Vt[i,:N]) for i in range(X.shape[0])])
  
  if norm_cols: # reverse the scaling
    M *= stdev[...,None,:]
    
  #define basis weights to return - just first N vectors
  if ret_eigenvals: W = Vt[...,:N,:].swapaxes(-2,-1) # don't scale W vectors if eigenvals are returned
  else: W = (Vt[...,:N,:] * d[...,:N,None]).swapaxes(-2,-1) # multiply W by the eigenvalues before returning just so they're consistent with sysrem/etc
  U = U[...,:N] # slice out the correct number of components
  
  if add_bias and mean_sub: # add bias term to the U vector
    s = 1/np.sqrt(X.shape[-2]) # sqrt(1/N) required to make new vector normalised
    U = np.concatenate([np.ones([*U.shape[:-1],1])*s  ,U],axis=-1) # this works for both 2D + 3D arrays
    W = np.concatenate([Xmean[...,None]/s,W],axis=-1) # also rescale the W vectors to account for s
      
  if ret_eigenvals: return U,W,M+Xmean[...,None,:],d[...,:N]
  else: return U,W,M+Xmean[...,None,:]

def sysrem(X,Xe,N=1,norm=False,tol=1e-5,mean_sub=True,max_iter=100,min_iter=5,add_bias=False,verbose=False):
  """
  Run SysRem on a 2D [time x wl] or 3D array [order x time x wl] or equivalent
  
  X - data
  Xe - uncertainties (stdev)
  N - number of passes of SysRem to run
  tol - tolerance for convergence (in terms of delta chi2 of successive model fits)
  mean_sub - performs weighted mean subtraction from each column if True. This should
    always be set unless other average subtraction is performed
  max_iter - maximum number of iterations
  min_iter - minimum number of iterations
  add_bias - add a bias vector to returned basis vectors U if True
  verbose - print out diagnostic information
  
  U - basis vectors of shape [(order x) time x N] or [(order x) time x N+1] if add_bias
  M - the SysRem model (after mean has been added back on)
  
  """
  
  assert X.shape==Xe.shape, ValueError("data and uncertainties arrays must be same shape")
  
  #create empty arrays for storage
  U = np.zeros((*X.shape[:-1],N))
  W = np.zeros([X.shape[0],X.shape[2],N] if X.ndim==3 else [X.shape[1],N])
  M = np.zeros(X.shape)  
  
  #precompute the weights (1 / var) and get weighted mean over columns
  L = 1. / Xe**2 #pre-compute the 1 / squared errors - ie weights
  #replace any zero uncertainties - inf weights - with more sensible values
  L[np.isinf(L)] = 0.

  if mean_sub:
    Xmean = np.sum(L*X,axis=-2) / np.sum(L,axis=-2)
    Xmean[np.isnan(Xmean)] = 0. # reset nans to zero (caused by column of inf errors -> div by zero)
  else: Xmean = np.zeros([X.shape[0],X.shape[-1]] if X.ndim==3 else [X.shape[-1],])
  #create array of active residuals (weighted mean subtracted if set)
  R = X - Xmean[...,None,:]
  
  #identify columns with all inf uncertainties or std=0. These should be removed from the algorithm
  col_filt = np.all(np.isinf(Xe),axis=-2) + np.isclose(np.std(X,axis=-2),0)
  col_filt = ~col_filt # set to True for good columns
  
  if X.ndim==2: #if X is 2D
    for n in range(N): # loop over N passes of SysRem
      if verbose: print('running pass {} of {}'.format(n,N))
      u,w,m = sysrem_pass(R[...,col_filt]-M[...,col_filt],L[...,col_filt],tol=tol,norm=norm,max_iter=max_iter,min_iter=min_iter,verbose=verbose)      
      M[...,col_filt] += m #add sysrem passes to model
      U[:,n] = u #store basis vectors
      W[:,n][col_filt] = w #store basis vectors
  
  elif X.ndim==3: #if X is 3D  
    for order in range(X.shape[0]): # loop over each order
      print("performing sysrem for order {}".format(order+1))    
      for n in range(N): #loop over N passes of sysrem
        if verbose: print('running pass {} of {}'.format(n,N))
        u,w,m = sysrem_pass((R[order]-M[order])[...,col_filt[order]],L[order][...,col_filt[order]],tol=tol,norm=norm,max_iter=max_iter,min_iter=min_iter,verbose=verbose)
        M[order][...,col_filt[order]] += m #add sysrem passes to model  
        U[order,:,n] = u #store basis vectors
        W[order,:,n][col_filt[order]] = w #store basis vectors

  if add_bias and mean_sub: # add bias term to the U vector and W vector to account for mean subtraction
    s = 1/np.sqrt(X.shape[-2]) # sqrt(1/N) required to make new vector normalised
    U = np.concatenate([np.ones([*U.shape[:-1],1])*s  ,U],axis=-1) # this works for both 2D + 3D arrays
    W = np.concatenate([Xmean[...,None]/s,W],axis=-1) # also rescale the W vectors to account for s
  
  return U,W,M + Xmean[...,None,:]

def sysrem_pass(R,W,norm=False,tol=1e-5,max_iter=100,min_iter=5,verbose=False):
  """
  Run single pass of sysrem - returns the basis vector + model.
  This takes in the inverse variance as the weights.
  Should use sysrem function directly, which is a wrapper around this func.
  
  R - data to fit
  W - weights, ie 1 / Re**2 (inverse variance)
  norm - specify whether to normalise the vertical vector, so that diag(U.T @ U)=1
    (SysRem does not guarantee U are orthonormal, like for PCA)
  
  """
  
  #initiate initial guess for 'optical state parameter'
#  a = np.ones(R.shape[0]) # initiate as a const vector
  a = np.random.normal(0,1,R.shape[0]) # initiate as a random vector
  a = a / np.sqrt(np.dot(a,a)) # normalise the vector
  
  #calculate chi2
  chi2_old = (W * R**2).sum()
  
  RW = R * W #can pre-multiply this as it takes a while
  
  #loop over iterations
  for q in range(max_iter):
    #get coefficients for each light curve
    c = np.dot(a,RW) / np.dot(a**2,W) #much faster using matrix algebra
    c[np.isnan(c)] = 0. #set any nans to zero - can happen where whole rows/columns have inf uncertainties
    
    #get 'airmass' for all light curves
    a = np.dot(RW,c) / np.dot(W,c**2) #much faster using matrix algebra
    if norm: a = a / np.sqrt(np.dot(a,a)) # force the vertical vector to be normalised
    a[np.isnan(a)] = 0.

    #calcalate the chi2 of the model fit - can ignore first few until min_iter is reached
    if q > min_iter or verbose:
      M = np.outer(a,c) # compute the current model
      #get new residuals from fit
      res_new = R - M
      chi2_new = (W * res_new**2).sum() #calculate the chi2

    if verbose: #print some diagnostic info
      print(" sysrem_pass iteration {}: delta chi2 = {}".format(q,chi2_old - chi2_new))
    
    #break from the loop if converged
    if q > min_iter and (chi2_old - chi2_new) < tol:
      if verbose: print(' sysrem converged after {} iterations'.format(q))      
      break
            
    #if the loop continues, store the new chi2 value as the old one
    if q > min_iter or verbose: chi2_old = chi2_new

  else:
    print(' warning: sysrem didnt converge after {} iterations'.format(q+1))
  
  return a,c,M

