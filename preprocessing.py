
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
