import numpy as np
import pickle
import matplotlib
import pickle
import pyfftw
from math import pi as PI
from decimal import Decimal
from matplotlib import cm, rc
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)

try:
    import matplotlib.pyplot as plt
    
    ifplot = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    from matplotlib import ticker
    from pylab import axes
    try:
        import seaborn as sns
        
        sns.set(style='white')
        colors = [sns.xkcd_rgb['denim blue'], sns.xkcd_rgb['pale red'],
                  sns.xkcd_rgb['olive green'], sns.xkcd_rgb['golden yellow']]
    except:
        colors = ['b', 'r', 'g', 'y']
except:
    print 'Problem with matplotlib.'

matplotlib.rc('axes', labelsize=25) 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 
rc('text', usetex=True)

def initc(x):  # Initial condition
 
    u0 = np.zeros_like(x, dtype='longdouble')
    u0 = np.cos(x/L)*(1.+np.sin(x/L))

    return u0

def wavenum(Mx):  # Wavenumber evaluation in Fourier space

    kxx = np.fft.rfftfreq(Mx, 1./Mx).astype(float)
    kx = np.zeros_like(kxx, dtype='longdouble')
    kx = kxx/L

    return kx

def fftrtf(uspec):  # Replicate half of the variable for symmetry in Fourier space  

    rtf = pyfftw.empty_aligned((Mx,), dtype='clongdouble')
    usp = np.conjugate(uspec[::-1])
    uspect = np.delete(usp, [0,Mx//2], None)
    rtf = np.concatenate((uspec[:], uspect[:]), axis=0)

    return rtf

def weights(x):  # Spatial integration weights

    weights = np.empty_like(x, dtype='longdouble')
    dx = np.empty_like(x, dtype='longdouble')
    nx = len(x)
    for i in range(nx-1):
	    dx[i] = x[i+1] - x[i]

    dx = np.delete(dx, [len(x)-1], None)

    for j in range(nx):
        if j == 0:
            weights[j] = dx[0]/2
	elif j == nx-1:
	    weights[j] = dx[-1]/2
	else:
	    weights[j] = dx[j-1]/2 + dx[j]/2

    return weights

def antialias(uhat,vhat):  # Anti-aliasing using padding technique

    N = len(uhat)
    M = 2*N
    uhat_pad = np.concatenate((uhat[0:N/2], np.zeros((M-N,)), uhat[N/2:]), axis=0)
    vhat_pad = np.concatenate((vhat[0:N/2], np.zeros((M-N,)), vhat[N/2:]), axis=0)
    u_pad = pyfftw.interfaces.numpy_fft.ifft(uhat_pad)
    v_pad = pyfftw.interfaces.numpy_fft.ifft(vhat_pad)
    w_pad = u_pad*v_pad
    what_pad = pyfftw.interfaces.numpy_fft.fft(w_pad)
    what = 2.*np.concatenate((what_pad[0:N/2], what_pad[M-N/2:M]), axis=0)

    return what

def aalias(uhat):  # To calculate (uhat)^2 in real space and transform to Fourier space  

    ureal = pyfftw.interfaces.numpy_fft.irfft(uhat)
    nlt = ureal.real*ureal.real
    usp = pyfftw.interfaces.numpy_fft.rfft(nlt)

    return usp

def alias(uht):  # To calculate (uht)^2 in real space

    url = pyfftw.interfaces.numpy_fft.ifft(uht)
    nlter = url.real*url.real

    return nlter

def fwnum(Mx):

    alpha = np.fft.fftfreq(Mx, 1./Mx).astype(int)
    alpha[Mx//2] *= -1

    return alpha
 
def kssol(u0):  # Solver for start at time t = 0  

    Tf = Decimal("150.0")                       # Final time
    t = Decimal("0.0")				# Current time
    h = Decimal("0.25")
    dt = float(h)				# Size of each time step
    nt = int(Tf/h)  				# Number of time steps

    kx = wavenum(Mx)
           					
    A = np.ones((Mx//2+1,)) 
    k2 = -(kx**2)+(kx**4)
    k2 += (c*A)
          
    u = pyfftw.empty_aligned((Mx//2+1,nt+1), dtype='clongdouble')  # Variable in Fourier space
    us0 = pyfftw.empty_aligned((Mx//2+1,), dtype='clongdouble')
    u[:,0] = pyfftw.interfaces.numpy_fft.rfft(u0)
    u[0,0] -= u[0,0]
    nlin = pyfftw.empty_aligned((Mx//2+1,nt+1), dtype='clongdouble')  
    nlin[:,0] = aalias(u[:,0])
    nlinspec = pyfftw.empty_aligned((Mx//2+1,nt+1), dtype='clongdouble')  # Non-linear term in Fourier space
    nlinspec[:,0] = -0.5*1j*kx*nlin[:,0]
    nls = pyfftw.empty_aligned((Mx//2+1,), dtype='clongdouble')
    nlspec = pyfftw.empty_aligned((Mx//2+1,), dtype='clongdouble')
    nondx = pyfftw.empty_aligned((Mx,nt+1), dtype='longdouble')
    nondx2 = pyfftw.empty_aligned((Mx,nt+1), dtype='longdouble')
    nondx[:,0] = alias(1j*fwnum(Mx)*fftrtf(u[:,0]))
    nondx2[:,0] = alias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,0]))
    ur = pyfftw.empty_aligned((Mx,nt+1), dtype='longdouble')  # Variable in real space
    ur[:,0] = pyfftw.interfaces.numpy_fft.irfft(u[:,0]).real
    wt = weights(x)        
    en = np.empty((nt+1,), dtype='longdouble')  # Energy calculation
    en[0] = np.dot(wt, ur[:,0]*ur[:,0])
    ent = np.empty((nt+1,), dtype='longdouble')  # Time-derivative of energy dE/dt
    ent[0] = (2.*np.dot(wt, nondx[:,0]))-(2.*np.dot(wt, nondx2[:,0]))
    
    for i in range(nt):
        t += h
        print t
        if i==0:
            us0 = (u[:,i] + (dt*nlinspec[:,i]) + (dt*c*u[:,i]))/(A + (dt*(k2+A)))
            us0[0] -= us0[0]
            us0[-1] -= us0[-1]
            nls[:] = aalias(0.5*(u[:,0]+us0))
            nlspec[:] = -0.5*1j*kx*nls[:]
            u[:,i+1] = (u[:,i] - (0.5*dt*k2*u[:,i]) + (dt*nlspec[:]) + (0.5*dt*c*(u[:,i]+us0)))/(A + (0.5*dt*(k2+A)))
            u[0,i+1] -= u[0,i+1]
            u[-1,i+1] -= u[-1,i+1]
            ur[:,i+1] = pyfftw.interfaces.numpy_fft.irfft(u[:,i+1]).real
            en[i+1] = np.dot(wt, ur[:,i+1]*ur[:,i+1])
            nondx[:,i+1] = alias(1j*fwnum(Mx)*fftrtf(u[:,i+1]))
            nondx2[:,i+1] = alias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1]))
            #nondx[:,i] = pyfftw.interfaces.numpy_fft.ifft(antialias(1j*fwnum(Mx)*fftrtf(u[:,i+1]),1j*fwnum(Mx)*fftrtf(u[:,i+1])))
            #nondx2[:,i] = pyfftw.interfaces.numpy_fft.ifft(antialias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1]),-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1])))
            ent[i+1] = (2.*np.dot(wt, nondx[:,i+1]))-(2.*np.dot(wt, nondx2[:,i+1])) 
        elif i==1:
            nlin[:,i] = aalias(u[:,i])
            nlinspec[:,i] = -0.5*1j*kx*nlin[:,i]
            u[:,i+1] = ((4*u[:,i]) - u[:,i-1] + (4*dt*nlinspec[:,i]) +(4*dt*c*u[:,i]) - (2*dt*nlinspec[:,i-1]) - (2*dt*c*u[:,i-1]))/((3*A) + (2*dt*(k2+A)))
            u[0,i+1] -= u[0,i+1]
            u[-1,i+1] -= u[-1,i+1]
            ur[:,i+1] = pyfftw.interfaces.numpy_fft.irfft(u[:,i+1]).real
            en[i+1] = np.dot(wt, ur[:,i+1]*ur[:,i+1]) 
            nondx[:,i+1] = alias(1j*fwnum(Mx)*fftrtf(u[:,i+1]))
            nondx2[:,i+1] = alias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1])) 
            #nondx[:,i] = pyfftw.interfaces.numpy_fft.ifft(antialias(1j*fwnum(Mx)*fftrtf(u[:,i+1]),1j*fwnum(Mx)*fftrtf(u[:,i+1])))
            #nondx2[:,i] = pyfftw.interfaces.numpy_fft.ifft(antialias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1]),-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1])))
            ent[i+1] = (2.*np.dot(wt, nondx[:,i+1]))-(2.*np.dot(wt, nondx2[:,i+1]))          
        else:
            nlin[:,i] = aalias(u[:,i])
            nlinspec[:,i] = -0.5*1j*kx*nlin[:,i]
            u[:,i+1] = ((18*u[:,i]) - (9*u[:,i-1]) + (2*u[:,i-2]) + (18*dt*nlinspec[:,i]) + (18*dt*c*u[:,i]) - (18*dt*nlinspec[:,i-1]) - (18*dt*c*u[:,i-1]) + (6*dt*nlinspec[:,i-2]) + (6*dt*c*u[:,i-2]))/((11*A) + (6*dt*(k2+A)))
            u[0,i+1] -= u[0,i+1]
            u[-1,i+1] -= u[-1,i+1]
            ur[:,i+1] = pyfftw.interfaces.numpy_fft.irfft(u[:,i+1]).real   
            en[i+1] = np.dot(wt, ur[:,i+1]*ur[:,i+1])
            nondx[:,i+1] = alias(1j*fwnum(Mx)*fftrtf(u[:,i+1]))
            nondx2[:,i+1] = alias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1])) 
            #nondx[:,i] = pyfftw.interfaces.numpy_fft.ifft(antialias(1j*fwnum(Mx)*fftrtf(u[:,i+1]),1j*fwnum(Mx)*fftrtf(u[:,i+1])))
            #nondx2[:,i] = pyfftw.interfaces.numpy_fft.ifft(antialias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1]),-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+1])))
            ent[i+1] = (2.*np.dot(wt, nondx[:,i+1]))-(2.*np.dot(wt, nondx2[:,i+1]))

    return u, ur, en, ent


def kssol2(u1,u2,u3):  # Solver for restart at any time step  

    Tf = Decimal("150.0")
    h = Decimal("0.25")
    dt = float(h)
    nt = int(Tf/h)

    kx = wavenum(Mx)

    u = pyfftw.empty_aligned((Mx//2+1,nt+3), dtype='clongdouble')
    u[:,0] = u1
    u[:,1] = u2
    u[:,2] = u3
    ur = pyfftw.empty_aligned((Mx,nt+1), dtype='longdouble')
    ur[:,0] = pyfftw.interfaces.numpy_fft.irfft(u[:,2]).real 

    nlin = pyfftw.empty_aligned((Mx//2+1,nt+3), dtype='clongdouble')
    nlin[:,0] = aalias(u[:,0])
    nlin[:,1] = aalias(u[:,1])
    nlin[:,2] = aalias(u[:,2])
    nlinspec = pyfftw.empty_aligned((Mx//2+1,nt+3), dtype='clongdouble')
    nlinspec[:,0] = -0.5*1j*kx*nlin[:,0]
    nlinspec[:,1] = -0.5*1j*kx*nlin[:,1]
    nlinspec[:,2] = -0.5*1j*kx*nlin[:,2]

    nondx = pyfftw.empty_aligned((Mx,nt+1), dtype='longdouble')
    nondx2 = pyfftw.empty_aligned((Mx,nt+1), dtype='longdouble')
    nondx[:,0] = alias(1j*fwnum(Mx)*fftrtf(u[:,2]))
    nondx2[:,0] = alias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,2]))
             						
    A = np.ones((Mx//2+1,)) 
    k2 = -(kx**2)+(kx**4)
    k2 += (c*A)        

    wt = weights(x)        
    en = np.empty((nt+1,), dtype='longdouble')
    en[0] = np.dot(wt, ur[:,0]*ur[:,0])
    ent = np.empty((nt+1,), dtype='longdouble')
    ent[0] = (2.*np.dot(wt, nondx[:,0]))-(2.*nu*np.dot(wt, nondx2[:,0]))
    
    for i in range(nt):
        print i
        u[:,i+1] = ((18*u[:,i]) - (9*u[:,i-1]) + (2*u[:,i-2]) + (18*dt*nlinspec[:,i]) +(18*dt*c*u[:,i]) - (18*dt*nlinspec[:,i-1]) - (18*dt*c*u[:,i-1]) + (6*dt*nlinspec[:,i-2]) + (6*dt*c*u[:,i-2]))/((11*A) + (6*dt*k2))
        u[0,i+3] -= u[0,i+3]
        u[-1,i+3] -= u[-1,i+3]
        #u[:,i+3].real -= u[:,i+3].real
        ur[:,i+1] = pyfftw.interfaces.numpy_fft.irfft(u[:,i+3]).real
        en[i+1] = np.dot(wt, ur[:,i+1]*ur[:,i+1])
        nlin[:,i+3] = aalias(u[:,i+3])
        nlinspec[:,i+3] = -0.5*1j*kx*nlin[:,i+3]
        nondx[:,i+1] = alias(1j*fwnum(Mx)*fftrtf(u[:,i+3]))
        nondx2[:,i+1] = alias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+3])) 
        #nondx[:,i] = pyfftw.interfaces.numpy_fft.ifft(antialias(1j*fwnum(Mx)*fftrtf(u[:,i+3]),1j*fwnum(Mx)*fftrtf(u[:,i+3]))) 
        #nondx2[:,i] = pyfftw.interfaces.numpy_fft.ifft(antialias(-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+3]),-1.*((fwnum(Mx))**2)*fftrtf(u[:,i+3]))) 
        ent[i+1] = (2.*np.dot(wt, nondx[:,i+1]))-(2.*nu*np.dot(wt, nondx2[:,i+1]))       

    return u, ur, en, ent        
          									

Mx = 2**7							
L = 16
dx = (2.*L*PI)/np.float(Mx)                    							
c = 0.               						
x = np.arange(0., Mx)*dx
u0 = initc(x)
u, ur, en, ent = kssol(u0)

    
