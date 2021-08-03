import argparse
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#%matplotlib inline
pylab.rcParams['figure.figsize'] = 12, 8  # that's default image size for this interactive session

class SAR:
    def __init__(self, _fc=1.0e10, _PRF=500, _Tp=6.033e-6, _alpha7s=4.0e12, _fs=30e6, _R0=7500, _V=200, _L=1.0, _Azfq=500):
        self.fc = _fc                     # radar beam carrier frequency (Hz)
        self.c  = 3.0e8                   # light speed (speed of wave) (m/s)
        self.lambd = self.c/self.fc       # wavelength of carrier = c/fc (m)
        self.A0 = 1.0                     # amplitude of transimtter signal (m)
        self.PRF = _PRF                   # pulse repetition frequency (Hz)
        self.Tp = _Tp                     # pulse duration (s)
        self.alpha7s = _alpha7s           # LFM pulse chirp rate (Range FM rate) (Hz/s)
        self.tot_samples = 271            # number of time samples total
        self.tot_az_samples = 1024        # number of time samples total
        self.fs = _fs                     # range sampling frequency
        self.R0 = _R0                     # closest range of the target R0 (m)
        self.B = self.alpha7s*self.Tp     # range bandwidth B = alpha * Tp (Hz)
        self.V = _V                       # radar moving speed (m/s)
        self.AFMrate= -2*self.V*self.V/(self.R0*self.lambd) # azimuth FM rate = -2V*V/(R0*lambda) (Hz/s)
        self.L = _L                      # antenna length (m)
        self.Rs = self.c/self.fs          # slant range sample spacing Rs = c/fs (m)
        self.thetaH = self.lambd/self.L   # Radar 3-db beamwidth thetaH = lambda / L (radians)
        self.As=self.V/self.PRF           # azimuth sample spacing As = V/PRF (m)
        self.Ls=self.R0*self.thetaH       # synthetic apreture lentgh Ls = R0 thetaH (m)
        self.y1 = self.Ls/2                    # Center of Ls
        self.Bdop= 2*self.V/self.L        # Doppler frequency bandwidth Bdop = 2V/L (Hz)
        self.Naz = int(1.1*550)           # number of azimuth samples 
        self.Nr = self.fs*self.Tp         # number of time samples within LFM pulse 
        self.R1 = self.R0/math.cos(0.5 * self.thetaH) # stant range (m) 
        self.R3 = self.R1        
        self.Ta = self.Ls/self.V          # azimuth samples
        self.Azfq = _Azfq                 # azimuth frequency..
        print(" Ls : ", self.Ls, "As : ", self.As, " AFMrate: ", self.AFMrate, ", thetaH: ", self.thetaH, " B: ", self.B, )
        
    def point_target_simulation(self, m=0, n=0):
        Ttot = float(self.tot_samples) / self.fs
        rb = np.zeros((self.Naz, self.tot_samples)).astype(np.complex64)
        ui = 0
        for i in range(self.Naz):
            tn = np.linspace(0, Ttot, self.tot_samples)
            if i < m or i > m + self.Naz:
                rb[i] = tn * 0.0
            else:
                mask = abs((tn-self.Tp/2)/self.Tp)<=0.5
                thau_ui = 2 * math.sqrt((ui -  self.y1)**2 + self.R0**2) / self.c
                thau_0 = 2 * math.sqrt(self.R0**2) / self.c
                rb[i] = np.exp(-1j * 2* np.pi * self.fc * thau_ui + 1j * np.pi * self.alpha7s * (tn-(self.Tp/2) - (thau_ui - thau_0 ) )**2) * mask
            ui = ui + self.As
        return rb 
    

    def baseband_signal(self):
        # baseband signal....................................
        Ttot = float(self.tot_samples) / self.fs
        t = np.linspace(0, Ttot, self.tot_samples)
        mask = abs((t-self.Tp/2)/self.Tp)<=0.5
        mask = mask.astype(int)
        p = (np.exp(1j*np.pi*self.alpha7s*(t-self.Tp/2)**2))*mask
        return p

    def range_matched_filter(self):
        Ttot = float(self.tot_samples) / self.fs
        t = np.linspace(0, Ttot, self.tot_samples)
        mask = abs((t-self.Tp/2)/self.Tp)<=0.5
        mask = mask.astype(int)
        hr = np.exp(-1j * np.pi * self.alpha7s * (t-self.Tp/2)**2) * mask
        return hr


    def azimuth_matched_filter(self):
        Ttot = float(self.tot_az_samples) / self.Azfq
        # change my dim
        s = np.linspace(0, Ttot, self.tot_az_samples)
        beta7s = -2 * self.V**2 / (self.lambd * self.R0)
        mask = abs((s-self.Ta/2)/self.Ta)<=0.5
        mask = mask.astype(int)
        #azimuth Ttot :   2048
        #azimuth Ta :   1.125
        #azimuth beta7s :  -355.55555555555554
        print( "azimuth Ttot :  ", Ttot)
        print( "azimuth Ta :  ", self.Ta)
        print(" azimuth beta7s : ", beta7s )

        ha = np.exp(-1j * np.pi * beta7s * (s-self.Ta/2)**2)*mask
        return ha

    def range_compression(self, rb):
        rb_fq = np.fft.fftshift(np.fft.fft(rb))
        hr = self.range_matched_filter()
        hr_fq = np.fft.fftshift(np.fft.fft(hr))
        print("rb_fq.shape :: ", rb_fq.shape)
        print("hr_fq.shape :: ", hr_fq.shape)
        res = rb_fq * hr_fq
        res = (np.fft.ifft(np.fft.fftshift(res)))  
        return res

    def azimuth_compression(self, rb):
        rb_new = np.zeros((self.tot_az_samples, self.tot_samples), dtype = 'complex_')
        rb_fq = np.fft.fftshift(np.fft.fft(rb))
        ha = self.azimuth_matched_filter()
        ha_fq = np.fft.fftshift(np.fft.fft(ha)) 
        print("rb_fq.shape :: ", rb_fq.shape)
        print("ha_fq.shape :: ", ha_fq.shape)
  
        for i in range(rb_fq.shape[1]):
            res1 = np.fft.fftshift(np.fft.fft(rb[:,i], 1024)) * ha_fq
            res = (np.fft.ifft(np.fft.fftshift(res1)))
            rb_new[:,i] = res   
        
        return rb_new


def plot_reflected_pulses(rb):
    # two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, sharex=True)
    (M, N) = rb.shape
    x = range(0, N)
    for i in range(M):
        y1 = rb[i].real*10 + i
        y2 = rb[i].imag*10 + i  
        # try commenting these two lines to get a nice visualization of all the pulses
        if i % 30 != 0:
           continue
   
        axarr[0].plot(x, y1)
        axarr[1].plot(x, y2)
        
    axarr[0].set_title('reflected real part ')
    axarr[1].set_title('reflected imaginary part ')

    axarr[1].set_xlabel('time samples')
    axarr[0].set_ylabel('azimuth samples')
    axarr[1].set_ylabel('azimuth samples')
    axarr[1].set_ylim([-10,M])
    axarr[0].set_ylim([-10,M])
    axarr[0].set_xlim([0,N])
      
def plot_pulses_3D(p, title, scale=1):
    # two subplots, the axes array is 1-d
    
    (M, N) = p.shape
    X = np.arange(0, N, 1)
    Y = np.arange(0, M, 1)

    X, Y = np.meshgrid(X, Y)
    Z_list =[[0 for x in range(N)]for y in range(M)]
    Z = np.array(Z_list)

    for i in range(M):
        for j in range(N):
            Z[i][j] = abs(p[i][j].real)/scale
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.1, antialiased=False)      
    ax.set_zlim(0,101)
    fig.colorbar(surf)
    plt.title(title)


def main(args):
    sar = SAR()
    
    rb = sar.point_target_simulation()

    plot_reflected_pulses(rb)

    range_compressed_rb = sar.range_compression(rb)

    plot_pulses_3D(range_compressed_rb, 'range compressed', 1)

    azimuth_compressed_rb = sar.azimuth_compression(range_compressed_rb)

    plot_pulses_3D(azimuth_compressed_rb, 'azimuth compressed', 100)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAR simulation')
    parser.add_argument('--fc', type=int, default=1.0e10)
    args = parser.parse_args()

    main(args)  
    
    
