### https://github.com/rdireen/scikit-rf/blob/50f0e13646472848f92427df64ab764d3de3b1fb/doc/source/tutorials/LNA%20Example.ipynb#L23
### Let's design a LNA using Infineon's BFU520 transistor.
### First we need to import scikit-rf and a bunch of other utilities:
### https://leleivre.com/rf_noise_temperature.html
### https://leleivre.com/rf_gammatoz.html
### 
import numpy as np

import skrf
from skrf.media import DistributedCircuit
import skrf.frequency as freq
import skrf.network as net
import skrf.util
#from quantiphy import Quantity

import matplotlib.pyplot as plt

#%matplotlib inline
plt.rcParams['figure.figsize'] = [10, 10]

###************************************************###
### User defined input                             ###
###************************************************###
f = freq.Frequency(0.4, 2, 101, 'ghz')
tem = DistributedCircuit(f, z0=50)
fuser = 915e+6 # frequency of interest for user in Hz
#s2p_filename = 'BFU725F_2V_5mA_S_N.s2p'
s2p_filename = 'BFU520_05V0_010mA_NF_SP.s2p'
###************************************************###

# import the scattering parameters/noise data for the transistor
#bjt = net.Network(s2p_filename)
#f = freq.Frequency(bjt.frequency.start, bjt.frequency.stop, 201, "hz") 
bjt = net.Network(s2p_filename).interpolate(f)

print("#######################################")
print( bjt )
print("#######################################")
# Let's plot the smith chart for it:
fig, ax = plt.subplots()
plt.title("Fig. 1 Deice's s-parameters")
bjt.plot_s_smith( ax=ax, draw_labels=True )


# Now let's calculate the source and load stablity curves.
# I'm slightly misusing the Network type to plot the curves;
# normally the curves you pass in to Network should be a function of frequency,
# but it also works to draw these circles as long as you don't try to use
# any other functions on them.
sqabs = lambda x: np.square(np.absolute(x))

delta = bjt.s11.s*bjt.s22.s - bjt.s12.s*bjt.s21.s
rl = np.absolute((bjt.s12.s * bjt.s21.s)/(sqabs(bjt.s22.s) - sqabs(delta)))
cl = np.conj(bjt.s22.s - delta*np.conj(bjt.s11.s))/(sqabs(bjt.s22.s) - sqabs(delta))

rs = np.absolute((bjt.s12.s * bjt.s21.s)/(sqabs(bjt.s11.s) - sqabs(delta)))
cs = np.conj(bjt.s11.s - delta*np.conj(bjt.s22.s))/(sqabs(bjt.s11.s) - sqabs(delta))

def calc_circle(c, r):
    theta = np.linspace(0, 2*np.pi, 1000)
    return c + r*np.exp(1.0j*theta)

fig, ax = plt.subplots()
plt.title("Fig. 2 Source stability circles")
for i, f in enumerate(bjt.f):
    # decimate it a little
    if i % 10 != 0:
        continue
    n = net.Network(name=str(f/1.e+9), s=calc_circle(cs[i][0, 0], rs[i][0, 0]))
    n.plot_s_smith(m=0, n=0, ax=ax, draw_labels=True)
    n.name

fig, ax = plt.subplots()
plt.title("Fig. 3 Load stability circles")
for i, f in enumerate(bjt.f):
    # decimate it a little
    if i % 10 != 0:
        continue
    n = net.Network(name=str(f/1.e+9), s=calc_circle(cl[i][0, 0], rl[i][0, 0]))
    n.plot_s_smith(m=0, n=0, ax=ax, draw_labels=True)
    n.name

# So we can see that we need to avoid inductive loads near short circuit in the input matching
# network and high impedance inductive loads on the output.
# Let's draw some constant noise circles. First we grab the noise parameters
# for our target frequency from the network model:
idx_fuser = skrf.util.find_nearest_index(bjt.f, fuser)

# we need the normalized equivalent noise and optimum source coefficient
# to calculate the constant noise circles
rn = bjt.rn[idx_fuser]/50
gamma_opt = bjt.g_opt[idx_fuser]
# lets calculate some interesting bjt parameters
fmin = bjt.nfmin[idx_fuser] # is it in dB ???, probably noise factor
fmin_dB = 10 * np.log10(fmin)
max_gain = 20 * np.log10(bjt.max_gain[idx_fuser])
max_stable_gain = 20 * np.log10(bjt.max_stable_gain[idx_fuser])

fig, ax = plt.subplots()
plt.title("Fig. 4 Constant noise circles")
for nf_added in [0, 0.1, 0.2, 0.3]:
    nf = 10**(nf_added/10) * fmin
    N = (nf - fmin)*abs(1+gamma_opt)**2/(4*rn)
    c_n = gamma_opt/(1+N)
    r_n = 1/(1-N)*np.sqrt(N**2 + N*(1-abs(gamma_opt)**2))
    n = net.Network(name=str(nf_added), s=calc_circle(c_n, r_n))
    n.plot_s_smith(m=0, n=0, ax=ax, draw_labels=True)


# z_s = net.s2z(np.array([[[gamma_opt]]]))[0,0,0] # old way of calculating it
# bjt.z_opt[idx_fuser] # yet better way than below of finding z_s optimum
z_s = skrf.Gamma0_2_zl(Gamma=gamma_opt, z0=50)[0]
g_s_mag = skrf.mathFunctions.complex_2_magnitude(gamma_opt) # just checking if it is the same as in s2p file
g_s_ang = skrf.mathFunctions.complex_2_degree(gamma_opt) # just checking if it is the same as in s2p file
S11 = 20 * np.log10(skrf.mathFunctions.complex_2_magnitude(gamma_opt))
### Calculate noise temperature
Tref = 290 # reference temp in Kelvin which equals to 16.85deg Celsius
NTK = Tref * ( 10**(fmin/10)-1 ) # Noise temperature in Kelvin
#######
# Another way of calculating real and img part of impedance from gamma_opt
# MAG = skrf.mathFunctions.complex_2_magnitude(gamma_opt)
# ANG = skrf.mathFunctions.complex_2_degree(gamma_opt)
# PI = np.pi
# Z0 = 50
# REAL=(Z0*(1-(MAG*MAG)))/(1+(MAG*MAG)-(2*MAG*np.cos((ANG/360)*2*PI)))
# IMAG=(2*MAG*np.sin((ANG/360)*2*PI)*Z0)/(1+(MAG*MAG)-(2*MAG*np.cos((ANG/360)*2*PI)))
#######
print("#####################################################")
print("#########         INPUT                    ##########")
print("Network name:          ", bjt.name)
print("Network frequency:     ", bjt.frequency)
print("Network Z0:            ", bjt.z0[0][0], "ohm")
print("Freq user [Hz]       = ", format(fuser, "0.0f"), "Hz")
print("NFmin [dB]           = ", format(fmin_dB, "0.3f"), "dB")
print("Noise Temp.          = ", format(NTK, "0.3f"), "Kelvin")
print("Gain max [dB]        = ", format(max_gain, "0.3f"), "dB")
print("Gain stable max [dB] = ", format(max_stable_gain, "0.3f"), "dB")
print("K -stab. factor      = ", format(bjt.stability[idx_fuser], ".3f"))
print("Gamma_opt            = ", format(gamma_opt, ".3f"))
print("Z_opt [ohm]          = ", format(z_s, ".3f"), "ohm")
print("S11 [dB]             = ", format(S11, "0.1f"), "dB")
print("gamma_opt_mag        = ", format(g_s_mag, ".3f"))
print("gamma_opt_ang        = ", format(g_s_ang, ".3f"), "deg")
print("#####################################################")
# **********
gamma_s = 0.0

gamma_l = np.conj(bjt.s22.s - bjt.s21.s*gamma_s*bjt.s12.s/(1-bjt.s11.s*gamma_s))
gamma_l = gamma_l[idx_fuser, 0, 0]
is_gamma_l_stable = np.absolute(gamma_l - cl[idx_fuser]) > rl[idx_fuser]

gamma_l, is_gamma_l_stable[0,0]

def calc_matching_network_vals(z1, z2):
    flipped = np.real(z1) < np.real(z2)
    if flipped:
        z2, z1 = z1, z2
        
    # cancel out the imaginary parts of both input and output impedances    
    z1_par = 0.0
    z1_par = 1e-6 # Am I missing something? Zero above causes divide by zero error
    if abs(np.imag(z1)) > 1e-6:
        # parallel something to cancel out the imaginary part of
        # z1's impedance
        z1_par = 1/(-1j*np.imag(1/z1))
        z1 = 1/(1./z1 + 1/z1_par)
    z2_ser = 0.0
    z2_ser = 1e-6 # Am I missing something? Zero above causes divide by zero error
    if abs(np.imag(z2)) > 1e-6:
        z2_ser = -1j*np.imag(z2)
        z2 = z2 + z2_ser
        
    Q = np.sqrt((np.real(z1) - np.real(z2))/np.real(z2))
    x1 = -1.j * np.real(z1)/Q
    x2 = 1.j * np.real(z2)*Q
    
    x1_tot = 1/(1/z1_par + 1/x1)
    x2_tot = z2_ser + x2
    if flipped:
        return x2_tot, x1_tot
    else:
        return x1_tot, x2_tot

z_l = net.s2z(np.array([[[gamma_l]]]))[0,0,0]
# note that we're matching against the conjugate;
# this is because we want to see z_l from the BJT side
# if we plugged in z the matching network would make
# the 50 ohms look like np.conj(z) to match against it, so
# we use np.conj(z_l) so that it'll look like z_l from the BJT's side
# I am not sure what calc_matching_network_vals() does
# I think it returns L-matching network with first element being a parallel
# and second being a series element.
# Does skrf have built in functions to calculate matching networks and
# componnet values, so functions like calc_matching_network_vals() 
# would not be needed?
z_par, z_ser = calc_matching_network_vals(np.conj(z_l), 50)
z_l, z_par, z_ser

# *****
# Let's calculate what the component values are:

c_par = np.real(1/(2j*np.pi*915e+6*z_par))
l_ser = np.real(z_ser/(2j*np.pi*915e+6))

c_par, l_ser

# The capacitance is kind of low but the inductance seems reasonable. 
# Let's test it out:
fig, ax = plt.subplots()
plt.title("Fig. 5 Output matching")
output_network = tem.shunt_capacitor(c_par) ** tem.inductor(l_ser)
amplifier = bjt ** output_network
amplifier.plot_s_smith()

#That looks pretty reasonable; let's take a look at the S21 to see what we got:
fig, ax = plt.subplots()
plt.title("Fig. 6 Amplifier gain [dB]")
amplifier.s21.plot_s_db() # how to plot a nice grid and log frequency axis?

# So about 18 dB gain; let's see what our noise figure is:

NF_dB = 10*np.log10(amplifier.nf(50.)[idx_fuser])

# So 0.96 dB NF, which is reasonably close to the BJT tombstone optimal NF of 0.95 dB
################

print("#####################################################")
print("#########         LOAD                     ##########")
print("Network name:              ", bjt.name)
print("Network frequency:         ", bjt.frequency)
print("Network Z0:                ", bjt.z0[0][0], "ohm")
print("Freq user [Hz]:            ", format(fuser, "0.0f"), "Hz")
print("Optimum gamma load:        ", format(gamma_l, ".3f") )
print("gamma load stable:         ", is_gamma_l_stable[0,0] )
print("NFmin [dB]:                ", format(fmin_dB, "0.3f"), "dB")
print("Amplifier NF [dB]:         ", format(NF_dB , "0.3f"), "dB")
print("Amplifier gain [dB]:       ", 0 ) # how to print it at freq of interest [idx_fuser]
print("Load impedance [ohm]:      ", format( z_l , ".3f") )
print("Parallel impedance [ohm]:  ", format(z_par, ".2f"), "ohm")
print("Series impedance [ohm]:    ", format(z_ser, ".2f"), "ohm")
print("Parallel capacitance [F]:  ", c_par )
print("Series iinductor [H]:      ", l_ser )
print("#####################################################")