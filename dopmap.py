"""
Translated from dopmap.R (R -> Python).
Depends on lobe.py for drawing lobe/streams if desired.

Example:
fig, ax = dopmap.dopmap("dopmap_demo_tumen.dat")
dopmap.draw_2ndlobe(ax, q=0.455, incl=52, porb=0.117, m1=0.785, color="white")
dopmap.draw_stream(ax, q=0.455, incl=52, porb=0.117, m1=0.785, color="white")
# plt.savefig("dopmap_demo_tumen.pdf")
plt.draw()
plt.show()

fig, ax, img = dopmap.trail("trail_demo_tumen.dat", value='model', nphase=20, nvel=100)
# plt.savefig("trail_demo_tumen.pdf")
plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# import lobe helpers optionally
try:
    from lobe import *
except Exception:
    pass


def rw_colors(n, bf=1.0):
    if n <= 0:
        return []
    x = np.linspace(0.0, 1.0, n)
    r = 1.0 / (1.0 + np.exp((0.375 - x) / 0.06))
    g = 1.0 / (1.0 + np.exp((0.70 - x) / 0.05))
    # b uses a gaussian component and a logistic component
    gauss = (1.0 / np.sqrt(2 * np.pi) / 0.15) * np.exp(-0.5 * ((x - 0.25) / 0.15) ** 2)
    gauss = gauss / gauss.max()
    b = np.maximum(bf * gauss, 2.0 / (1.0 + np.exp((1.0 - x) / 0.05)))
    cols = [ (ri, gi, bi) for ri, gi, bi in zip(r, g, b) ]
    return ListedColormap(cols)


def dopmap(file, ax=None, **imshow_kwargs):
    d = np.loadtxt(file, skiprows=1)
    imgd = int(np.sqrt(d.shape[0]))
    mapdata = d[:, 2].reshape((imgd, imgd))
    vmin = d[0, 0]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    cmap = rw_colors(512)
    x = np.linspace(vmin, -vmin, imgd)
    y = np.linspace(vmin, -vmin, imgd)
    ax.imshow(mapdata, extent=(x[0], x[-1], y[0], y[-1]),
              origin='lower', cmap=cmap, aspect='equal', zorder=0,
              **imshow_kwargs)
    ax.set_xlabel('Vx (km/s)')
    ax.set_ylabel('Vy (km/s)')
    print(f"min={mapdata.min()}, max={mapdata.max()}")
    return fig, ax

def trail(file, value='data', nphase=20, nvel=50, cmap='gray_r', wrap_phase=True, flip_phase=True, show=False):
    """
    Trailed spectrum plot with flexible phase handling.

    Parameters
    ----------
    file : str
        Input file path. Columns: v, sp0, sp1, phase
    value : str
        'data', 'model', or 'residual'
    nphase : int
        Number of bins along phase
    nvel : int
        Number of bins along velocity
    cmap : str
        Matplotlib colormap
    wrap_phase : bool
        If True, phase values >= 1 are wrapped (ph - 1)
    flip_phase : bool
        If True, phase 0 is at the bottom of the plot
    show : bool
        Whether to call plt.show() at the end
    """
    data = np.loadtxt(file, skiprows=1)
    v = data[:,0]
    sp0 = data[:,1]
    sp1 = data[:,2]
    ph = data[:,3]

    if wrap_phase:
        ph = ph % 1.0

    # Decide which value to use
    if value == 'data':
        ydata = sp0
    elif value == 'model':
        ydata = sp1
    elif value == 'residual':
        ydata = sp1 - sp0
    else:
        raise ValueError("value must be 'data', 'model', or 'residual'")

    # Phase bins
    ph_min, ph_max = ph.min(), ph.max()
    ph_bins = np.linspace(ph_min, ph_max, nphase+1)
    ph_centers = 0.5*(ph_bins[:-1] + ph_bins[1:])

    # Velocity bins
    v_min, v_max = v.min(), v.max()
    v_bins = np.linspace(v_min, v_max, nvel+1)
    v_centers = 0.5*(v_bins[:-1] + v_bins[1:])

    # Prepare 2D array
    img = np.full((nphase, nvel), np.nan)

    # Bin the data
    for i in range(nphase):
        ph_mask = (ph >= ph_bins[i]) & (ph < ph_bins[i+1])
        if not np.any(ph_mask):
            continue
        for j in range(nvel):
            v_mask = (v >= v_bins[j]) & (v < v_bins[j+1])
            mask = ph_mask & v_mask
            if np.any(mask):
                img[i,j] = np.mean(ydata[mask])

    if flip_phase:
        img = img[::-1,:]  # flip vertical axis so phase=0 at bottom
        ph_min, ph_max = ph_max, ph_min

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(img, extent=(v_min, v_max, ph_min, ph_max),
                   origin='lower', aspect='auto', cmap=cmap)
    ax.set_xlabel('Radial velocity (km/s)')
    ax.set_ylabel('Orbital phase')
    plt.colorbar(im, ax=ax, label=value)
    if show:
        plt.show()
    return fig, ax, img
