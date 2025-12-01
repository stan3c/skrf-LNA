# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:43:39 2021

@author: Przem
"""
# ## This script is inspired by rdireen but rewritten from scratch
# ## to generate a comprehensive report about a device used to build an LNA.
# ## https://github.com/rdireen/scikit-rf/blob/50f0e13646472848f92427df64ab764d3de3b1fb/doc/source/tutorials/LNA%20Example.ipynb#L23
# ## Let's design a LNA using Infineon's BFU520 transistor.
# ## First we need to import scikit-rf and a bunch of other utilities:
# ## https://leleivre.com/rf_noise_temperature.html
# ## https://leleivre.com/rf_gammatoz.html
# ##
import numpy as np

import skrf
from skrf.media import DistributedCircuit
import skrf.frequency as freq
import skrf.network as net
import skrf.util
# from quantiphy import Quantity

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# %matplotlib inline
plt.rcParams['figure.figsize'] = [10, 10]

# ##================================================###
# ## User defined input                             ###
# ##================================================###
f = freq.Frequency(0.4, 2, 101, 'ghz')
Z0 = 50
tem = DistributedCircuit(f, z0=Z0)
fuser = 900e+6  # frequency of interest for user in Hz
s2p_filename = 'BFU725F_2V_5mA_S_N.s2p'
# s2p_filename = 'BFU520_05V0_010mA_NF_SP.s2p'
plot_inline = False
plot_save = False
if plot_inline:
    print("Info: Images will be ploted inline.")
else:
    print("Info: Images will not be ploted inline.")
if plot_save:
    print("Info: Images will be saved to file.")
else:
    print("Info: Images will not be saved to file.")
# ##================================================###

#  IEEE style for Smith chart plots
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.transparent": True
})


# global counter for plots
_plot_counter = 0
def next_plot_number():
    global _plot_counter
    _plot_counter += 1
    return _plot_counter


def wrap_text(s, width=60, indent="\t\t"):
    result = []
    start = 0
    while start < len(s):
        if len(s) - start <= width:
            result.append(s[start:])
            break

        cut = start + width
        segment = s[start:cut]

        # priorytet: przecinek, kropka, średnik, dwukropek
        punct_pos = max(segment.rfind(','), segment.rfind('.'),
                        segment.rfind(';'), segment.rfind(':'))

        if punct_pos != -1:
            cut = start + punct_pos + 1
        else:
            space_pos = segment.rfind(' ')
            if space_pos != -1:
                cut = start + space_pos

        result.append(s[start:cut].strip())
        start = cut

    # dodaj wcięcie do drugiej i kolejnych linii
    for i in range(1, len(result)):
        result[i] = indent + result[i]

    return "\n".join(result)


def load_model(s2p_filename):
    # import the scattering parameters/noise data for the transistor, s2p file
    try:
        bjt = net.Network(s2p_filename).interpolate(f)
        print("="*60)
        print(wrap_text(str(bjt), 60))
        print("="*60)
        print(f"\n✓ Model file loaded: {s2p_filename}")
    except:
        print(f"\n✗ Error: Could not load the file {s2p_filename}")
        raise  # Stop the program

    fig, ax = plt.subplots(figsize=(3.15, 3.15))  # ~8 cm × 8 cm
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Fig. {next_plot_number()} Deice's s-parameters", pad=6)
    # bjt.plot_s_smith( ax=ax, draw_labels=True )

    # 1) Draw the background of the Smith chart (grid, unit circle)
    # Use the function from scikit-rf: smith() or Smith()
    # Depending on the version of scikit-rf:
    try:
        # New versions
        skrf.plotting.smith(ax=ax, draw_labels=True)
    except AttributeError:
        # Older versions: auxiliary Network method (fallback)
        bjt.plot_s_smith(ax=ax, draw_labels=True)
        # Note: if this call has already drawn the curves,
        # the grid may be default;
        # in older versions the smith-grid is sometimes drawn automatically.

    # 2) Draw each curve with its own marker (readable in B/W)
    # Add custom axis labels
    # Define styles: different colors + markers
    styles = [
        {"label": r"$S_{11}$", "color": "tab:blue",   "marker": "o"},
        {"label": r"$S_{21}$", "color": "tab:orange", "marker": "s"},
        {"label": r"$S_{12}$", "color": "tab:green",  "marker": "^"},
        {"label": r"$S_{22}$", "color": "tab:red",    "marker": "D"},
    ]

    curves = [
        bjt.s[:, 0, 0],  # S11
        bjt.s[:, 1, 0],  # S21
        bjt.s[:, 0, 1],  # S12
        bjt.s[:, 1, 1],  # S22
    ]

    # Plot each curve with its style
    for curve, st in zip(curves, styles):
        ax.plot(curve.real, curve.imag,
                color=st["color"],
                marker=st["marker"],
                markevery=max(1, len(curve)//5),
                mec='black', mew=0.6, mfc='white',
                label=st["label"])

    # Axis labels and legend
    ax.set_xlabel(r"Re($\Gamma$)")
    ax.set_ylabel(r"Im($\Gamma$)")
    ax.legend(
        loc='center left',
        bbox_to_anchor=(0.7, -0.1),   # put the legend box outside the chart
        frameon=False,
        ncol=2,
        fontsize=7,                # smaller font
        handlelength=1.5,          # short lines in the legend
        handletextpad=0.4,         # marker to text spacing
        markerscale=0.8,           # smaller markers in the legend
        labelspacing=0.3           # short spacing between legend lines
            )

    # axis limits
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    # axis labels
    ax.set_xlabel(r"Re($\Gamma$)", labelpad=5)
    ax.set_ylabel(r"Im($\Gamma$)", labelpad=10)

    plt.tight_layout()
    plt.show()
    return bjt


def calc_circle(c, r):
    theta = np.linspace(0, 2*np.pi, 1000)
    return c + r*np.exp(1.0j*theta)


def sl_stability(bjt):
    # Now let's calculate the source and load stablity curves.
    # I'm slightly misusing the Network type to plot the curves;
    # normally the curves you pass in to Network should be a function
    # of frequency, but it also works to draw these circles as long
    # as you don't try to use any other functions on them.

    sqabs = lambda x: np.square(np.absolute(x))

    delta = bjt.s11.s*bjt.s22.s - bjt.s12.s*bjt.s21.s
    rl = np.absolute((bjt.s12.s * bjt.s21.s)/(sqabs(bjt.s22.s) - sqabs(delta)))
    cl = (delta * np.conj(bjt.s11.s) - bjt.s22.s) / (sqabs(bjt.s22.s)
                                                     - sqabs(delta))

    rs = np.absolute((bjt.s12.s * bjt.s21.s)/(sqabs(bjt.s11.s) - sqabs(delta)))
    cs = np.conj(delta*np.conj(bjt.s22.s) - bjt.s11.s)/(sqabs(bjt.s11.s)
                                                        - sqabs(delta))
    if plot_inline:
        fig, ax = plt.subplots()
        plt.title(f"Fig. {next_plot_number()} Source stability circles")
        for i, f in enumerate(bjt.f):
            # decimate it a little
            if i % 10 != 0:
                continue
            n = net.Network(name=str(f/1.e+9), s=calc_circle(cs[i][0, 0],
                                                             rs[i][0, 0]))
            n.plot_s_smith(m=0, n=0, ax=ax, draw_labels=True, chart_type='y')
            n.name
            # Ograniczenia osi
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

        # Etykiety osi
        ax.set_xlabel(r"Re($\Gamma$)", labelpad=5)
        ax.set_ylabel(r"Im($\Gamma$)", labelpad=10)

        if plot_save:
            plt.savefig(f"Fig. {_plot_counter} Source stability circles",
                        dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.title(f"Fig. {next_plot_number()} Load stability circles")

        for i, f in enumerate(bjt.f):
            # decimate it a little
            if i % 10 != 0:
                continue
            n = net.Network(name=str(f/1.e+9), s=calc_circle(cl[i][0, 0],
                                                             rl[i][0, 0]))
            n.plot_s_smith(m=0, n=0, ax=ax, draw_labels=True, chart_type='z')
            n.name
            # Ograniczenia osi
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

        # Etykiety osi
        ax.set_xlabel(r"Re($\Gamma$)", labelpad=5)
        ax.set_ylabel(r"Im($\Gamma$)", labelpad=10)

        if plot_save:
            plt.savefig(f"Fig. {_plot_counter} Load stability circles",
                        dpi=300, bbox_inches='tight')

    return rl, cl, rs, cs


def plot_noise_circles(bjt, fuser, filename="noise_circles.png"):
    has_noise = hasattr(bjt, 'rn') and bjt.rn is not None
    if has_noise is True:

        # Let's draw some constant noise circles. First we grab the noise
        # parameters for our target frequency from the network model:
        idx_fuser = skrf.util.find_nearest_index(bjt.f, fuser)
        # we need the normalized equivalent noise and optimum source
        # coefficient to calculate the constant noise circles
        rn = bjt.rn[idx_fuser] / 50.0       # normalized Rn
        gamma_opt = bjt.g_opt[idx_fuser]    # optimum source reflection
        fmin = bjt.nfmin[idx_fuser]         # min. noise factor (linear)
        NFmin_dB = 10 * np.log10(fmin)
        max_gain = 20 * np.log10(bjt.max_gain[idx_fuser])
        max_stable_gain = 20 * np.log10(bjt.max_stable_gain[idx_fuser])

        # Prepare Smith chart
        fig, ax = plt.subplots(figsize=(3.15, 3.15))
        skrf.plotting.smith(ax=ax, draw_labels=True, chart_type='z')
        ax.set_title(f"Fig. {next_plot_number()} Constant Noise Circles", pad=6)
        ax.set_xlabel(r"Re($\Gamma$)", labelpad=8)
        ax.set_ylabel(r"Im($\Gamma$)", labelpad=8)

        centers = []
        radii = []
        markers = ['o', 's', '^', 'x']
        colors = plt.cm.tab10.colors
        # rysowanie kilku okręgów dla różnych przyrostów NF
        # NF numbers are NFmin + number from the list  - FIX ME
        for i, (nf_added, marker) in enumerate(zip([0, 0.1, 0.2, 0.3], markers)):
            nf = 10**(nf_added/10) * fmin   # noise factor dla danego przyrostu
            N = (nf - fmin) * abs(1+gamma_opt)**2 / (4*rn)
            c_n = gamma_opt / (1+N)
            r_n = 1/(1-N) * np.sqrt(N**2 + N*(1-abs(gamma_opt)**2))
    
            centers.append(c_n)
            radii.append(r_n)
    
            # dodaj okrąg na wykres
            circle = plt.Circle((c_n.real, c_n.imag), r_n,
                                edgecolor=colors[i], facecolor='none',
                                lw=0.8, alpha=0.7)
                     #           label=f"ΔNF={nf_added:.1f} dB")
            ax.add_patch(circle)

            # marker w środku okręgu
            ax.scatter(centers[0].real, centers[0].imag, marker=marker,
                       color='black', s=30)
                      # label=f"ΔNF={nf_added:.1f} dB")

            # marker na obwodzie okręgu wzdłuż linii Im(Γ)=0.5
            y_line = 0.1

            # środek i promień okręgu
            xc, yc = c_n.real, c_n.imag
            r = r_n

            # kąt dla punktu "na lewo od środka"
            theta = np.pi
    
            # dodaj lekkie przesunięcie w górę (np. 0.1 rad ~ 6°)
            theta_shift = theta - 0.2

            # współrzędne markera
            marker_x = xc + r * np.cos(theta_shift)
            marker_y = yc + r * np.sin(theta_shift)
    
            ax.scatter(marker_x, marker_y, marker=marker, color=colors[i], s=30,
                       label=f"ΔNF={nf_added:.1f} dB")

        c0 = centers[0]
        # obliczenie impedancji w omach
        Z0 = 50.0
        z_norm = (1 + c0) / (1 - c0)
        Z = z_norm * Z0
        # dodanie wartości impedancji obok markera
        ax.text(c0.real-0.06, c0.imag+0.06,
                f"{Z.real:.0f} + j{Z.imag:.0f} Ω",
                fontsize=7, color="black")

        # z_s = net.s2z(np.array([[[gamma_opt]]]))[0,0,0] # old way
        # z_s = skrf.Gamma0_2_zl(Gamma=gamma_opt, z0=50)[0]
        z_s = bjt.z_opt[idx_fuser]  # the best way of finding z_s optimum
        # Calculating noise gamma magnitude and angle just to confirm
        # the model works and returns the same nr that is in s2p model file
        g_s_mag = skrf.mathFunctions.complex_2_magnitude(gamma_opt)
        g_s_ang = skrf.mathFunctions.complex_2_degree(gamma_opt)
        S11dB = 20 * np.log10(skrf.mathFunctions.complex_2_magnitude(gamma_opt))
        # ## Calculate noise temperature
        Tref = 290  # reference temp in Kelvin which equals to 16.85deg Celsius
        NTK = Tref * ( 10**(fmin/10)-1 )  # Noise temperature in Kelvin

        # Transistor's S-parameters at the user frequency
        S11 = bjt.s[idx_fuser, 0, 0]
        S12 = bjt.s[idx_fuser, 0, 1]
        S21 = bjt.s[idx_fuser, 1, 0]
        S22 = bjt.s[idx_fuser, 1, 1]
        # Checking stability
        K = (1 - np.abs(S11)**2 - np.abs(S22)**2 + np.abs(S11*S22 - S12*S21)**2) / (2 * np.abs(S12*S21))
        Delta = S11*S22 - S12*S21
        is_stable = K > 1 and np.abs(Delta) < 1

        print(f"\nS-parameters @ {fuser/1e6:.2f} MHz:")
        print(f"  S11  = {np.abs(S11):.3f}∠{np.angle(S11, deg=True):.1f}° ({20*np.log10(np.abs(S11)):.2f} dB)")
        print(f"  S21  = {np.abs(S21):.3f}∠{np.angle(S21, deg=True):.1f}° ({20*np.log10(np.abs(S21)):.2f} dB)")
        print(f"  S12  = {np.abs(S12):.3f}∠{np.angle(S12, deg=True):.1f}°")
        print(f"  S22  = {np.abs(S22):.3f}∠{np.angle(S22, deg=True):.1f}°")

        print("\nStability analysis:")
        print(f"  K = {K:.3f}")
        print(f"  |Δ| = {np.abs(Delta):.3f}")
        print(f"  Status: {'STABLE' if is_stable else 'UNSTABLE (needs to be stabilized)'}")
        print("\n✓ Noise parameters loaded.")
        print(f"\nGamma opt (min. noise) @ {fuser/1e6:.2f} MHz:")
        print(f"  S11  = {g_s_mag:.3f}∠{g_s_ang:.1f}° ({20*np.log10(np.abs(gamma_opt)):.2f} dB)")
        print(f"  Zopt = {z_s:.1f} Ω")
        print("")
        print("="*60)
        print("INPUT")
        print("="*60)
        print("Network name           ", bjt.name)
        print("Network frequency    = ", bjt.frequency)
        print("Network Z0           = ", bjt.z0[0][0], "ohm")
        print("Freq user [Hz]       = ", format(fuser, "0.0f"), "Hz")
        print("Z_opt [ohm]          = ", format(z_s, ".1f"), "Ω")
        print("NFmin [dB]           = ", format(NFmin_dB, "0.3f"), "dB")
        print("Noise Temp.          = ", format(NTK, "0.3f"), "Kelvin")
        print("Gamma_opt            = ", format(gamma_opt, ".3f"))
        print("gamma_opt_mag        = ", format(g_s_mag, ".3f"))
        print("gamma_opt_ang        = ", format(g_s_ang, ".3f"), "deg")
        print("Gain max [dB]        = ", format(max_gain, "0.3f"), "dB")
        print("Gain stable max [dB] = ", format(max_stable_gain, "0.3f"), "dB")
        print("K -stab. factor      = ", format(bjt.stability[idx_fuser],
                                                ".3f"))
        print("S11 [dB]             = ", format(S11dB, "0.1f"), "dB")
        print("."*60)

        if plot_inline:
            # legenda w prawym dolnym rogu, lekko poniżej wykresu
            ax.legend(loc='lower right', bbox_to_anchor=(1.0, -0.25),
                      ncol=1, frameon=False, fontsize=7)

            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            plt.tight_layout()
            plt.show()
            if plot_save:
                plt.savefig('lna_noise_circles.png', dpi=300,
                            bbox_inches='tight')

    else:
        print("\n✗ Model has no noise data.")
        print("   Noise charts will not be generated.")


def calc_matching_network_vals(z1, z2, f, idx_fuser, z0=50, topology='L'):
    """
    Oblicza sieć dopasowującą pomiędzy dwiema impedancjami.
    
    Parameters:
    -----------
    z1 : complex
        Impedancja źródłowa (Ω)
    z2 : complex  
        Impedancja docelowa (Ω)
    frequency : rf.Frequency
        Obiekt częstotliwości scikit-rf
    z0 : float
        Impedancja charakterystyczna systemu (domyślnie 50 Ω)
    topology : str
        Topologia: 'L' (L-network), 'Pi', 'T'
        
    Returns:
    --------
    network : rf.Network
        Sieć dopasowująca jako obiekt scikit-rf Network
    components : dict
        Słownik z wartościami komponentów
    """
    
    # Konwersja impedancji na moduł i fazę
    R1 = np.real(z1)
    X1 = np.imag(z1)
    R2 = np.real(z2)
    X2 = np.imag(z2)
    
    print(f"\nDopasowanie: Z1 = {R1:.2f} + j{X1:.2f} Ω -> Z2 = {R2:.2f} + j{X2:.2f} Ω")
    
    # Częstotliwość centralna
    f_center = f.f[idx_fuser]
    omega = 2 * np.pi * f_center
    
    # Tworzenie media
    media = skrf.media.DefinedGammaZ0(frequency=f, z0=z0)
    
    if topology == 'L':
        # Topologia L-network (najprostsza, 2 elementy)
        # Decyzja: która impedancja jest niższa?
        
        if R1 < R2:
            # R1 jest niższe -> L szeregowo od z1, C równolegle do z2
            # Kompensacja reaktancji wejściowej
            X_comp = -X1
            
            # Obliczenia dla sieci L
            Q = np.sqrt(R2/R1 - 1)
            X_series = Q * R1  # Reaktancja indukcyjna szeregowa
            X_shunt = R2 / Q   # Reaktancja pojemnościowa równoległa
            
            L_series = (X_series + X_comp) / omega
            C_shunt = 1 / (omega * X_shunt)
            
            # Sprawdzenie czy wartości są fizycznie realizowalne
            if L_series < 0:
                print(f"  ⚠ Ujemna indukcyjność! Dodaję kompensację kondensatorem...")
                C_series = -1 / (omega * (X_series + X_comp))
                L_series = 0
                series_element = media.capacitor(C_series)
                print(f"  C_series = {C_series*1e12:.2f} pF")
            else:
                series_element = media.inductor(L_series)
                print(f"  L_series = {L_series*1e9:.2f} nH")
            
            shunt_element = media.shunt_capacitor(C_shunt)
            print(f"  C_shunt = {C_shunt*1e12:.2f} pF")
            
            # Połączenie: element szeregowy ** element równoległy
            network = series_element ** shunt_element
            
            components = {
                'topology': 'L-network (series L/C, shunt C)',
                'L_series_nH': L_series * 1e9 if L_series > 0 else 0,
                'C_series_pF': C_series * 1e12 if L_series <= 0 else 0,
                'C_shunt_pF': C_shunt * 1e12,
                'Q': Q
            }
            
        else:
            # R2 jest niższe -> C szeregowo od z1, L równolegle do z2
            X_comp = -X1
            
            Q = np.sqrt(R1/R2 - 1)
            X_series = -R1 / Q  # Reaktancja pojemnościowa szeregowa (ujemna)
            X_shunt = Q * R2    # Reaktancja indukcyjna równoległa
            
            C_series = -1 / (omega * (X_series + X_comp))
            L_shunt = X_shunt / omega
            
            # Sprawdzenie realizowalności
            if C_series < 0:
                print(f"  ⚠ Ujemna pojemność! Dodaję kompensację indukcyjnością...")
                L_series = -(X_series + X_comp) / omega
                C_series = 0
                series_element = media.inductor(L_series)
                print(f"  L_series = {L_series*1e9:.2f} nH")
            else:
                series_element = media.capacitor(C_series)
                print(f"  C_series = {C_series*1e12:.2f} pF")
            
            shunt_element = media.shunt_inductor(L_shunt)
            print(f"  L_shunt = {L_shunt*1e9:.2f} nH")
            
            network = series_element ** shunt_element
            
            components = {
                'topology': 'L-network (series C, shunt L)',
                'C_series_pF': C_series * 1e12 if C_series > 0 else 0,
                'L_series_nH': L_series * 1e9 if C_series <= 0 else 0,
                'L_shunt_nH': L_shunt * 1e9,
                'Q': Q
            }
    
    else:
        raise NotImplementedError(f"Topologia '{topology}' nie jest jeszcze zaimplementowana")
    
    return network, components


def shunt_resistor_as_2port(R, frequency, z0=50):
    """
    Tworzy 2-portowy Network reprezentujący rezystor shunt do masy
    (między węzłem wyjściowym a masą) o wartości R.

    Parameters
    ----------
    R : float
        Rezystancja w omach (Ω).
    frequency : rf.Frequency
        Oś częstotliwości używana przez główny układ (np. bjt.frequency).
    z0 : float or array-like
        Impedancja odniesienia portów.

    Returns
    -------
    skrf.Network
        2-port do skaskadowania z układem (ABCD: [[1,0],[1/R,1]]).
    """
    npts = len(frequency.f)
    gamma = (R - z0) / (R + z0)

    s = np.zeros((npts, 2, 2), dtype=complex)
    s[:, 0, 0] = gamma   # S11
    s[:, 1, 1] = gamma   # S22
    s[:, 0, 1] = 0       # brak sprzężenia
    s[:, 1, 0] = 0

    R_shunt = skrf.Network(frequency=frequency, s=s, z0=z0)
    return R_shunt


def add_output_shunt_resistor(bjt, R_shunt, z0=50):
    """
    Dodaje rezystor shunt na wyjściu BJT (port 2 -> masa)
    poprzez kaskadę 2-portów.

    Parameters
    ----------
    bjt : rf.Network
        2-portowy Network tranzystora.
    R_shunt : float
        Wartość rezystora shunt (Ω) do masy na wyjściu.
    z0 : float or array-like
        Impedancja odniesienia portów.

    Returns
    -------
    rf.Network
        Skaskadowany układ: bjt ** shunt(R).
    """
    shunt_net = shunt_resistor_as_2port(R_shunt, bjt.frequency, z0=z0)
    return bjt ** shunt_net


def plot_gain_ieee(network, port_in=1, port_out=2, Z0=50,
                   title="Forward gain versus frequency",
                   label="|S21| gain",
                   savepath=None):
    """
    Rysuje wzmocnienie |S21| w dB w funkcji częstotliwości
    w stylu zgodnym z wymaganiami IEEE.

    Parameters
    ----------
    network : skrf.Network
        Obiekt z parametrami S (np. bjt).
    port_in : int
        Numer portu wejściowego (domyślnie 1).
    port_out : int
        Numer portu wyjściowego (domyślnie 2).
    Z0 : float
        Impedancja odniesienia (domyślnie 50 Ω).
    title : str
        Tytuł wykresu.
    label : str
        Etykieta dla legendy.
    savepath : str or None
        Jeśli podano ścieżkę, zapisuje wykres do pliku PNG.
    """

    # --- IEEE-style plotting defaults ---
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
    })

    # --- Figure sized for IEEE single-column (≈3.5 in wide) ---
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Plot selected S-parameter in dB
    network.s[:, port_out-1, port_in-1]  # just to confirm indexing
    getattr(network, f"s{port_out}{port_in}").plot_s_db(ax=ax, label=label)

    # Format x-axis to GHz
    def hz_to_ghz(x, pos):
        return f"{x/1e9:.2f}"
    ax.xaxis.set_major_formatter(FuncFormatter(hz_to_ghz))

    # Minor ticks and grid
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)

    # Axis labels and title
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title(title)

    # Legend
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath)

    plt.show()


# ============================================================================
# ============================================================================
# ============================================================================
bjt = load_model(s2p_filename)  # Load the model file and print the header
# Calculate source and load stability circles and plot them on a Smith chart
rl, cl, rs, cs = sl_stability(bjt)
plot_noise_circles(bjt, fuser, filename="noise_circles.png")
# ============================================================================
# Lets design a matching network
# Please specify in the configuration above whether matching
# should be done for S11 or for Gamma optimum (NFmin)
#
# note that we're matching against the conjugate;
# this is because we want to see z_s from the BJT side
# if we plugged in z the matching network would make
# the 50 ohms look like np.conj(z) to match against it, so
# we use np.conj(z_s) so that it'll look like z_s from the BJT's side
# Does skrf have built in functions to calculate matching networks and
# componnet values, so functions like calc_matching_network_vals()
# would not be needed?
idx_fuser = skrf.util.find_nearest_index(bjt.f, fuser)
# gamma_opt = bjt.g_opt[idx_fuser]    # optimum source reflection
# z_s = net.s2z(np.array([[[gamma_opt]]]))[0, 0, 0]
z_s = bjt.z_opt[idx_fuser]  # returns Z as seen from bjt towards input port


# 4. Calulate input matching network
matching_net, components = calc_matching_network_vals(Z0, np.conj(z_s),
                                                      f, idx_fuser, z0=Z0)

print("Z_opt = ", format(z_s, ".1f"), "Ω")
print("\n✓ Input matching network created!")
print(f"  Topology: {components['topology']}")
print(f"  Q = {components['Q']:.2f}")

# 5. POŁĄCZENIE SIECI: matching_network ** bjt
# Operator ** w scikit-rf oznacza kaskadowe połączenie (cascade)
lna_in_bjt = matching_net ** bjt

print("\n✓ Input Matching network successfully connected to BJT")
print(f"  Partial LNA (INPUT MATCHING + BJT): {lna_in_bjt}")

# S11 (dopasowanie wejściowe)
s11_matched = lna_in_bjt.s[idx_fuser, 0, 0]
print(f"\nResult @ {f.f[idx_fuser]/1e9:.2f} GHz:")
print(f"  S11 = {20*np.log10(np.abs(s11_matched)):.2f} dB")
print(f"  |Γ_in| = {np.abs(s11_matched):.3f}")

# S21 (wzmocnienie)
s21_matched = lna_in_bjt.s[idx_fuser, 1, 0]
print(f"  S21 = {20*np.log10(np.abs(s21_matched)):.2f} dB")


# ****************************************************************************
# OUTPUT MATCHING (not finished, mess)
# ****************************************************************************
gamma_s = 0.0
gamma_l = np.conj(bjt.s22.s - bjt.s21.s*gamma_s*bjt.s12.s/(1-bjt.s11.s*gamma_s))
gamma_l = gamma_l[idx_fuser, 0, 0]
is_gamma_l_stable = np.absolute(gamma_l - cl[idx_fuser]) > rl[idx_fuser]

# gamma_l, is_gamma_l_stable[0, 0]
# z_par, z_ser = calc_matching_network_vals(np.conj(z_l), 50)
# z_l, z_par, z_ser

# Transistor's S-parameters at the user frequency
S11 = lna_in_bjt.s[idx_fuser, 0, 0]
S12 = lna_in_bjt.s[idx_fuser, 0, 1]
S21 = lna_in_bjt.s[idx_fuser, 1, 0]
S22 = lna_in_bjt.s[idx_fuser, 1, 1]
z_l = lna_in_bjt.z[:, 1, 1][idx_fuser]
# Checking stability
K = (1 - np.abs(S11)**2 - np.abs(S22)**2 +
     np.abs(S11*S22 - S12*S21)**2) / (2 * np.abs(S12*S21))
Delta = S11*S22 - S12*S21
is_stable = K > 1 and np.abs(Delta) < 1

# We calculate output impedance of the bjt with input matching
# at design frequency f(idx_fuser), as seen towards the output port
# FIXME stability handling needs improvement
z_l = lna_in_bjt.z[:, 1, 1][idx_fuser]
if np.real(z_l) <= 0:  # Stability and negative Re need a better solution
    print("\n✗ Warning:")
    print("   Real part of output impedance is negative!")
    print("   Potential instability.")
    print("   Attempting to stabilize the network.")
    print("   Adding a shunt resistor at the output to stabilize network.")
    # Tworzenie media
    media = skrf.media.DefinedGammaZ0(frequency=lna_in_bjt.frequency, z0=Z0)
    for Rsh in range(500, 0, -50):  # ohm
        # print(Rsh)  # for debugging
        shunt_net = media.shunt_resistor(Rsh)
        # print(shunt_net)   # powinien być 2-Port Network  # for debugging
        # Połącz na wyjściu
        circuit = lna_in_bjt ** shunt_net

        # Transistor's S-parameters at the user frequency
        S11 = circuit.s[idx_fuser, 0, 0]
        S12 = circuit.s[idx_fuser, 0, 1]
        S21 = circuit.s[idx_fuser, 1, 0]
        S22 = circuit.s[idx_fuser, 1, 1]
        z_l = circuit.z[:, 1, 1][idx_fuser]
        # Checking stability
        K = (1 - np.abs(S11)**2 - np.abs(S22)**2 +
             np.abs(S11*S22 - S12*S21)**2) / (2 * np.abs(S12*S21))
        Delta = S11*S22 - S12*S21
        is_stable = K > 1 and np.abs(Delta) < 1
        if is_stable:
            lna_in_bjt = circuit
            print(f"\n✓ Stability achieved with Rshunt = {Rsh} Ω.")
            print("   Rshunt added at the output.")
            print(f"   Stabilized Z load: Zload = {np.real(z_l):.2f} + j{np.imag(z_l):.2f} Ω")
            break
if not is_stable and np.real(z_l) <= 0:
    print("\n✗ Error: Unable to stabilize the network!")
    print("   Real part of output impedance is negative!")
elif not is_stable:
    print("\n✗ Warning: Network (INPUT MATCHING + BJT) potentially unstable!")
    print("   Calculating output matching anyway.")

# Let's calculate what the component values are:
matching_net, components = calc_matching_network_vals(Z0, np.conj(z_l),
                                                      f, idx_fuser, z0=Z0)
lna_in_bjt_out = lna_in_bjt ** matching_net
print("\n✓ Output matching network created!")
print(f"  Topology: {components['topology']}")
print(f"  Q = {components['Q']:.2f}")

print("\n✓ Output Matching network successfully connected to BJT")
print(f"  Complete LNA (INPUT MATCHING + BJT + OUTPUT MATCHING): {lna_in_bjt_out}")

# S22 (output matching)
s22_matched = lna_in_bjt_out.s[idx_fuser, 0, 0]
print(f"\nResult @ {f.f[idx_fuser]/1e9:.2f} GHz:")
print(f"  S22 = {20*np.log10(np.abs(s22_matched)):.2f} dB")
print(f"  |Γ_in| = {np.abs(s22_matched):.3f}")

# S21 (wzmocnienie)
s21_matched = lna_in_bjt_out.s[idx_fuser, 1, 0]
print(f"  S21 = {20*np.log10(np.abs(s21_matched)):.2f} dB")

# c_par = np.real(1/(2j*np.pi*915e+6*z_par))
# l_ser = np.real(z_ser/(2j*np.pi*915e+6))

c_par = None
l_ser = None

for key, value in components.items():
    if key.startswith("C_") and value > 0:
        c_par = value
    elif key.startswith("L_") and value > 0:
        l_ser = value

# c_par = components["C_shunt_pF"]
# l_ser = components["L_series_nH"]

# The capacitance is kind of low but the inductance seems reasonable.
# Let's test it out:
fig, ax = plt.subplots()
plt.title("Fig. 5 Output matching")
output_network = tem.shunt_capacitor(c_par) ** tem.inductor(l_ser)
amplifier = bjt ** output_network
lna_in_bjt_out.plot_s_smith()

# That looks pretty reasonable; let's take a look at S21 to see what we got:
# fig, ax = plt.subplots()
# plt.title("Fig. 6 Amplifier gain [dB]")
# lna_in_bjt_out.s21.plot_s_db()
plot_gain_ieee(lna_in_bjt_out, title="BJT forward gain",
               label="S21 gain", savepath=None)
# So about 18 dB gain; let's see what our noise figure is:

NF_dB = 10*np.log10(lna_in_bjt_out.nf(50.)[idx_fuser])

# ###############

print("="*60)
print("OUTPUT")
print("="*60)
print("Network name:              ", bjt.name)
print("Network frequency:         ", bjt.frequency)
print("Network Z0:                ", bjt.z0[0][0], "ohm")
print("Freq user [Hz]:            ", format(fuser, "0.0f"), "Hz")
print("Optimum gamma load:        ", format(gamma_l, ".3f"))
#print("gamma load stable:         ", is_gamma_l_stable[0, 0])
print("Amplifier NF [dB]:         ", format(NF_dB, "0.3f"), "dB")
print("Amplifier gain @ fuser     ",
      format(float(amplifier.s21.s_db[idx_fuser]), "0.1f"), "dB")
print("Load impedance [ohm]:      ", format(z_l, ".3f"))
#print("Parallel impedance [ohm]:  ", format(z_par, ".2f"), "ohm")
#print("Series impedance [ohm]:    ", format(z_ser, ".2f"), "ohm")
print("Parallel capacitance [F]:  ", c_par)
print("Series iinductor [H]:      ", l_ser)
print("."*60)

print(f"Info: Generated a total of {_plot_counter} plots.")


def plot_source_stability(bjt):  # this function is wrong probably
    """
    Rysuje source stability circles dla modelu skrf.Network (bjt)
    na wykresie Smitha, z decymacją częstotliwości co 10.
    """

    # Styl IEEE
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
    })

    # Decymacja częstotliwości
    f = bjt.f[::10]
    S = bjt.s[::10, :, :]   # [freq, port, port]

    # Wyciągnięcie parametrów
    S11 = S[:,0,0]
    S12 = S[:,0,1]
    S21 = S[:,1,0]
    S22 = S[:,1,1]

    # Obliczanie source stability circles
    # Wzory: C_s = (D*conj(S22) - S11) / (|S22|^2 - |D|^2)
    #        r_s = |S12*S21| / (|S22|^2 - |D|^2)
    # gdzie D = S11*S22 - S12*S21
    D = S11*S22 - S12*S21
    C_s = (D*np.conj(S22) - S11) / (np.abs(S11)**2 - np.abs(D)**2)
    r_s = np.abs(S12*S21) / (np.abs(S11)**2 - np.abs(D)**2)

    # Rysowanie Smith chart
    fig, ax = plt.subplots(figsize=(3.15, 3.15))
    ax.set_aspect('equal', adjustable='box')
    plt.title("Fig. Source Stability Circles", pad=6)

    skrf.plotting.smith(ax=ax, draw_labels=True, chart_type	='y')
    ax.set_xlabel(r"Re($\Gamma$)", labelpad=8)
    ax.set_ylabel(r"Im($\Gamma$)", labelpad=8)

    # Rysowanie kół stabilności
    for c, r, freq in zip(C_s, r_s, f):
        circle = plt.Circle((c.real, c.imag), r,
                            edgecolor='tab:blue', facecolor='none',
                            lw=0.8, alpha=0.6)
        ax.add_patch(circle)
        # opcjonalnie podpis częstotliwości
        #ax.text(c.real+r+0.02, c.imag,
         #       f"{freq/1e9:.2f} GHz",
          #      fontsize=6, color='tab:blue')
    ax.scatter(C_s[0].real, C_s[0].imag, marker='o', color='black', s=25, label=f"f_min={f[0]/1e9:.2f} GHz")
    ax.scatter(C_s[-1].real, C_s[-1].imag, marker='s', color='black', s=25, label=f"f_max={f[-1]/1e9:.2f} GHz")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=7)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    plt.tight_layout()
    plt.show()


# ######
# Another way of calculating real and img part of impedance from gamma_opt
# MAG = skrf.mathFunctions.complex_2_magnitude(gamma_opt)
# ANG = skrf.mathFunctions.complex_2_degree(gamma_opt)
# PI = np.pi
# Z0 = 50
# REAL=(Z0*(1-(MAG*MAG)))/(1+(MAG*MAG)-(2*MAG*np.cos((ANG/360)*2*PI)))
# IMAG=(2*MAG*np.sin((ANG/360)*2*PI)*Z0)/(1+(MAG*MAG)-(2*MAG*np.cos((ANG/360)*2*PI)))
# ######


def old_calc_matching_network_vals(z1, z2):
    flipped = np.real(z1) < np.real(z2)
    if flipped:
        z2, z1 = z1, z2

    # cancel out the imaginary parts of both input and output impedance
    z1_par = 0.0
    # z1_par = 1e-6  # Am I missing something? Zero above causes divide by zero
    if abs(np.imag(z1)) > 1e-6:
        # parallel something to cancel out the imaginary part of
        # z1's impedance
        z1_par = 1/(-1j*np.imag(1/z1))
        z1 = 1/(1./z1 + 1/z1_par)
    z2_ser = 0.0
#    z2_ser = 1e-6  # Am I missing something? Zero above causes divide by zero
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


def old_stab_circles(bjt):
    fig, ax = plt.subplots()
    plt.title("Fig. 4 Constant noise circles")
    for nf_added in [0, 0.1, 0.2, 0.3]:
        nf = 10**(nf_added/10) * fmin
        N = (nf - fmin)*abs(1+gamma_opt)**2/(4*rn)
        c_n = gamma_opt/(1+N)
        r_n = 1/(1-N)*np.sqrt(N**2 + N*(1-abs(gamma_opt)**2))
        n = net.Network(name=str(nf_added), s=calc_circle(c_n, r_n))
        n.plot_s_smith(m=0, n=0, ax=ax, draw_labels=True)


