"""
A simple transmission spec model for the tutorial.

Neale Gibson
n.gibson@tcd.ie

"""

import numpy as np
import yaml
import os
import glob
from astropy import constants as c

class semianalytical(object):
  """
  Setup a simple transmission spec model using the semi-analytic model in Heng & Kitzmann
  (2017). The version here is the modified equations used in Gibson et al. (2020).
  
  To setup the model call:
  > model = semianalytical(wl,Rpl,P0,g,mmw,Rstar,temps,xs)
  wl - wavelength vector [A]
  Rpl - planet radius [m]
  P0 - reference pressure [bar]
  g - surface gravity [m/s^2]
  mmw - mean molecular weight [Da - ie proton mass]
  Rstar - stellar radius [m]
  temps - list of temperature arrays for cross-sections, ie [temps[0],temps[1],...]
  xs - list of cross-sections [each should be an array of (temps[n].size x wl.size) ]
  
  Once the class is initialised, you need to setup a parameter vector to call the model
  > pars = [temp,Pcloud,mix_h2,vel_broad,mixing_ratios x Nspecies]
  temp - temperature [K]
  Pcloud - cloud deck pressure [bar]
  mix_h2 - effective H2 VMR for Rayleigh scattering, does not impact scale height
  vel_broad - standard deviation of Gaussian broadening kernel [km/s] (if <= 1 not applied)
  vmrs - mixing ratio for each species
  
  """
  
  bar = 100000.
  u = 1.6605390666e-27
  k_B = 1.3806488e-23 #Boltzman const J/K
  gamma_EM = 0.57721 # Euler-Mascheroni constant

  def __init__(self,wl,R0,P0,g0,mmw,Rstar,temps,xs,n_kernel=50):
    
    self.wl = wl # Angstrom
    self.R0 = R0 # planet reference radius (m)
    self.P0 = P0 # reference pressure bar
    self.P0_SI = P0 * self.bar # reference pressure converted to SI
    self.g0 = g0 # surface grav SI
    self.mmw = 2.33 # mean molecular weight
    self.Rstar = Rstar # planet reference radius (m)
    self.k_ug = self.k_B / (np.mean(self.mmw) * self.u * self.g0) # k / ug for calculation of scale height
    
    #store cross-section information
    self.temperatures = temps
    self.xs = xs
    self.nspecies = len(xs)
    
    #some basic checks on inputs/etc
    if not len(self.temperatures)==len(self.xs): raise ValueError("temp arrays and xs arrays don't match")
    for i in range(self.nspecies):
      if not self.xs[i].shape[1]==self.wl.size: raise ValueError(f"xs[{i}] array not correct shape/size")    
    
    #get ray scatting cross-section - Dalgarno, A. and Williams, D. A. 1962
    self.xs_ray_h2 = (8.14e-13 / wl**4 + 1.28e-6 / wl**6 + 1.61 / wl**8) / 10000 #m^2
    
    #setup simple kernel for broadening
    res = np.mean(self.wl / np.gradient(self.wl)) # mean resolution, wl sampling should be close to uniform in v_space for this to work
    delta_vel = 299792458 / res / 1000 # km/s
    self.n_kernel = n_kernel
    self.v =  np.arange(-n_kernel,n_kernel+1) * delta_vel    
    self.exp_half_v2 =  np.exp(-0.5 * self.v**2)
    
  def tspec_r(self,pars):  

    #calculate constant for analytical model
    Hs = pars[0] * self.k_ug #(mean) scale height
    K = self.R0 + Hs * (self.gamma_EM + np.log(self.P0_SI / self.g0) + 0.5*np.log(2.*np.pi*self.R0/Hs) - np.log(self.mmw*self.u))
    
    #first get scattering cross-sections and store log for continuum model
    xs = pars[2] * self.xs_ray_h2 #first get Rayleigh scattering cross section
    log_scattering = np.log(xs)
    
    #loop over cross-sections to get summed cross-sections scaled by mixing ratios
    for i in range(self.nspecies): #add on remaining species using interpolators, weighted by abundance
      #interpolate cross-section to model temperature
      #print(pars[0],self.temperatures[i],np.arange(self.temperatures[i].size))
      frac_x0,x0 = np.modf(np.interp(pars[0],self.temperatures[i],np.arange(self.temperatures[i].size)))
      if x0==self.temperatures[i].size-1: # check upper boundaries
        frac_x0=1.0
        x0 = x0-1
      #print(frac_x0,x0)
      a,b = (1-frac_x0),frac_x0
      cross_sect = a*self.xs[i][int(x0)] + b*self.xs[i][int(x0)+1]
      xs += pars[4+i] * cross_sect
    
    #compute the model for provided species + scattering
    r_species = (K + Hs * np.log(xs))
    
    #calculate continuum model
    r_cont = (K + Hs * log_scattering)
    
    #get cloud deck altitude from pressure
    r_cloud = (Hs * np.log(self.P0/pars[1]) + self.R0)
  
    #truncate the continuum model and the full model with cloud deck
    self.r_cont = np.maximum(r_cont,r_cloud)    
    self.r = np.maximum(r_species,r_cloud)
    
    #convolve the model using simple broadening kernel
    #print(pars[3])
    if pars[3] > 1: # only apply if kernel width > 1
#      self.kernel = self.exp_half_v2 ** (1/pars[3]**2)
      self.kernel = np.exp(-0.5 /pars[3]**2 * self.v**2)
      self.kernel /= self.kernel.sum() # normalise
#      self.r = np.convolve(self.r,self.kernel / self.kernel.sum(),'same')
      self.r = np.convolve(np.pad(self.r,self.n_kernel,'edge'),self.kernel,'valid')
      #print("applying kernel")
    
    return self.r
  
  def tspec_f(self,*args): return (self.tspec_r(*args) / self.Rstar)**2
  def tspec_rprs(self,*args): return self.tspec_r(*args) / self.Rstar
  def tspec_nf(self,*args): return -(self.tspec_r(*args) / self.Rstar)**2
  def tspec_df(self,*args):
    r = self.tspec_r(*args)
    return (r / self.Rstar)**2 - (self.r_cont / self.Rstar)**2
  def tspec_ndf(self,*args): return -self.tspec_df(*args)
  
  def __call__(self,*args,**kwargs):
    """
    Define call function for class.
    
    """
    return self.tspec_r(*args,**kwargs)

### Below are functions to read in the line lists

#load data file
with open('xs_data.yml','r') as file:
  xs_config = yaml.safe_load(file)

def xs_DACE_multi(species,*args,**kwargs):
  """
  Wrapper for xs_DACE to loop over multiple species with same parameters
  
  """
  
  assert type(species) is list or type(species) is tuple, "species must be list or tuple of species"
  
  #create empty lists for outputs
  temps,pressures,xs = [],[],[]
  
  #loop over species and get data
  for sp in species:
    wl,t,p,x = xs_DACE(sp,*args,**kwargs)
    temps.append(t)
    pressures.append(p)
    xs.append(x)
  
  return wl,temps,pressures,xs

def xs_DACE(species,wl=None,wl_lims=[0,np.inf],wl_units='A',units='cm2',path=None,dtype=np.float32):
  """
  Simple function to read in DACE line lists
    
  """

  if path is None: path = xs_config['dace_path']
  dace_data = xs_config['dace_data']
  
  species_path = os.path.join(path,dace_data[species][0])
  print('getting {} opacities from path: {}'.format(species,species_path))
  
  filelist = glob.glob(os.path.join(species_path,'Out*.bin'))
  
  if not os.path.exists(species_path):
    raise ValueError('path {} does not exist'.format(species_path))
  if len(filelist)<1:
    raise ValueError('no files found in path {}'.format(species_path))
  
  #define empty lists for available temps and pressures
  temps = []
  pressures_str = []
  
  #loop through files and append each pressure and temperature string
  for i,file in enumerate(filelist):
    filename = file.split('/')[-1] #strip away directory
    temps.append(filename.split('_')[3]) #append temperature string
    pressures_str.append(filename.split('_')[-1].split('.')[0]) #pressure string
  stem = filename[:16] # Out_00000_42000_
    
  #sort and filter out duplicates
  temps = np.sort(np.array(list(set(temps)),dtype=int))
  #pressures a little more of a pain as stored as strings with p/n as positive/negative
  pressures_str = np.array(list(set(pressures_str)),dtype=str)
  log10_pressures = np.array([i.replace('p','+').replace('n','-') for i in pressures_str],dtype=int)/100  
  #print(log10_pressures,pressures_str)
  p_sort_ind = np.argsort(log10_pressures)
  pressures_str = pressures_str[p_sort_ind]
  pressures = 10**log10_pressures[p_sort_ind]
  
  #check correct number of temps and pressures found
  #should really be an assertion, but will leave for now
  #if temps.size * pressures.size == len(filelist): print('temp/pressure grid ok...')
  assert temps.size * pressures.size == len(filelist), "temp/pressure grid not correct shape"
    
  #get wl array in A
  mu_range = np.array(filename.split('_')[1:3],dtype=float)  
  nu = np.arange(*mu_range,0.01)
#  nu_inv = np.hstack([np.inf,1/nu[1:]])[::-1] # reverse direction of nu
  with np.errstate(divide='ignore'): nu_inv = 1/nu[::-1] # reverse direction of nu
  if wl_units=='A': wl_database = 1e8 * nu_inv
  elif wl_units=='nm': wl_database = 1e7 * nu_inv
  else: raise ValueError('wl_units not valid')
  
  if wl is not None:
    wl_lims[0],wl_lims[1] = wl.min(),wl.max() #overwrite if wl array explicitly given
    interpolate = True
    print('wl limits:',wl_lims)
  else: interpolate = False
  #print(wl_lims)
  wl_index = (wl_database > wl_lims[0]) * (wl_database < wl_lims[1]) #get index for wavelength limits
  #print(wl_index)
  wl_database = wl_database[wl_index]
  
  #loop over temps and pressures and get cross-sections in order
  if interpolate: xs = np.empty((len(temps),len(pressures),wl.size)) #use last file to set array shape
  else: xs = np.empty((len(temps),len(pressures),wl_database.size)) #or resize for interpolated grid
  for i,t in enumerate(temps):
    for j,p in enumerate(pressures_str):
      file = os.path.join(species_path,"{}{:05d}_{}.bin".format(stem,t,p))
      #print(' reading {}...'.format(file))
      if interpolate: xs[i,j] = np.interp(wl,wl_database,np.fromfile(file,dtype=np.float32)[::-1][wl_index])
      else: xs[i,j] = np.fromfile(file,dtype=np.float32)[::-1][wl_index]
  
  #correct the units of cross-sections if required
  if units=='cm2/g': pass
  elif units=='cm2': xs *= dace_data[species][1]*c.m_p.value*1000.
  elif units=='m2': xs *= dace_data[species][1]*c.m_p.value/10.
  else: print('units not valid'); return
  
  #pressures = pressures.astype(np.float) #convert pressures to float array
  #print(pressures,pressures_str)
  
  print('returning wl, pressure, temperature, xs with ranges:')
  if interpolate: print(' wl:     [{} -> {}]'.format(wl.min(),wl.max()))
  else: print(' wl:     [{} -> {}] {}'.format(wl_database.min(),wl_database.max(),wl_units))
  print(' temp:     [{} -> {}] K ({} samples)'.format(temps.min(),temps.max(),temps.size))
  print(' pressure: [{} -> {}] bar ({} samples)'.format(pressures.min(),pressures.max(),pressures.size))
  
  if interpolate: return wl,temps,pressures,xs.astype(dtype)
  else: return wl_database,temps,pressures,xs.astype(dtype)
