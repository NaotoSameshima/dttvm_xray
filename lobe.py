"""
Translated from lobe.R (R -> Python).
Provides functions to compute Roche lobes, stream trajectories and drawing helpers.
Author: translated for user
"""
import numpy as np
from math import pi, sin, cos, sqrt

# physical constants
gg = 6.667e-8
msun = 1.989e33


def rlq1(q):
    """Compute L1 Roche radius fraction (root of equation)."""
    if abs(1 - q) < 1e-4:
        return 0.5
    rl = 0.0
    rn = 1.0 - q
    # Newton iteration
    while True:
        rl = rn
        f = q / (1 - rl) ** 2 - 1 / rl ** 2 + (1 + q) * rl - q
        fa = 2 * q / (1 - rl) ** 3 + 2 / rl ** 3 + (1 + q)
        rn = rl - f / fa
        if abs(rl / rn - 1) <= 1e-4:
            break
    return rn


def pot(q, x, y, z):
    """Potential and radial derivative equivalent to R version.
    Returns dict with keys 'pot' and 'pr'.
    """
    r = np.sqrt(x * x + y * y + z * z)
    if abs(r) < 1e-10:
        raise ValueError("r ~ 0 in pot")
    rh = np.sqrt(x * x + y * y)
    st = rh / r
    if abs(rh) < 1e-10:
        cf = 1.0
    else:
        cf = x / rh
    r2 = 1.0 / (1.0 + q)
    r1 = np.sqrt(1 + r * r - 2 * r * cf * st)
    pot_val = -1.0 / r - 1.0 / q / r1 - 0.5 * (1.0 / q + 1.0) * (r2 * r2 + (r * st) ** 2 - 2 * r2 * r * cf * st)
    pr = 1.0 / r ** 2 + 1.0 / q / (r1 ** 3) * (r - cf * st) - 0.5 * (1.0 / q + 1.0) * 2 * (r * st * st - r2 * cf * st)
    return {'pot': pot_val, 'pr': pr}


def surf(q, rl1, nc):
    """Compute surface points used to generate lobe boundary."""
    dc = pi / nc
    ch = ((np.arange(1, nc + 1)) - 1) * pi / (nc - 1)
    rs1 = 1.0 - rl1
    fs = pot(q, rs1, 0.0, 0.0)
    rx = (1.0 - dc) * rs1
    r = [rs1]
    # discard first element of x,y arrays in original R; we will loop appropriately
    x = np.cos(ch)
    y = np.sin(ch)
    x = x[1:]
    y = y[1:]
    im = 20
    for i in range(len(x)):
        j = 0
        f = fs
        fpot = 1.0
        # iterate Newton-like
        while j < im and (abs(f['pot'] - fs['pot']) > 1e-4 or j == 0):
            j += 1
            r1 = rx
            f = pot(q, r1 * x[i], r1 * y[i], 0.0)
            rx = r1 - (f['pot'] - fs['pot']) / f['pr']
            if rx > rs1:
                rx = rs1
        r.append(rx)
    r = np.array(r)
    return {'r': r, 'ch': ch}


def lobes(q, rs, nc=72):
    l1 = surf(q, rs, nc)
    x1 = 1.0 - l1['r'] * np.cos(l1['ch'])
    y1 = -l1['r'] * np.sin(l1['ch'])
    # return closed polygon (concatenate reversed parts similar to R code)
    xs = np.concatenate([x1, x1[::-1]])
    ys = np.concatenate([y1, -y1[::-1]])
    return {'x': xs, 'y': ys}


def eqmot(z, w, z1, z2, qm):
    zr1 = z - z1
    zr2 = z - z2
    # Note: complex arithmetic; w and z may be complex
    wp = - (qm * zr2 / (abs(zr2) ** 3) + zr1 / (abs(zr1) ** 3)) / (1 + qm) - 2j * w + z
    zp = w
    return {'wp': wp, 'zp': zp}


def intrk(z, w, dt, z1, z2, qm):
    zx = z
    wx = w
    tmp = eqmot(zx, wx, z1, z2, qm)
    hz0 = tmp['zp'] * dt
    hw0 = tmp['wp'] * dt
    zx = z + hz0 / 2
    wx = w + hw0 / 2
    tmp = eqmot(zx, wx, z1, z2, qm)
    hz1 = tmp['zp'] * dt
    hw1 = tmp['wp'] * dt
    zx = z + hz1 / 2
    wx = w + hw1 / 2
    tmp = eqmot(zx, wx, z1, z2, qm)
    hz2 = tmp['zp'] * dt
    hw2 = tmp['wp'] * dt
    zx = z + hz2
    wx = w + hw2
    tmp = eqmot(zx, wx, z1, z2, qm)
    hz3 = tmp['zp'] * dt
    hw3 = tmp['wp'] * dt
    dz = (hz0 + 2 * hz1 + 2 * hz2 + hz3) / 6
    dw = (hw0 + 2 * hw1 + 2 * hw2 + hw3) / 6
    return {'dz': dz, 'dw': dw}


def stream(qm, rl1, nmax=250, rd=0.1):
    cm = qm / (1 + qm)
    z1 = -cm
    z2 = 1 - cm
    wm1 = np.conj(-1j * cm)
    eps = 1e-3
    z = (rl1 - cm - eps) + 0j
    w = 0.0 + 0j
    t = 0.0
    dt = 1e-4
    it = 0
    r = 1
    wout = []
    wkout = []
    while it < nmax:
        it += 1
        rk = intrk(z, w, dt, z1, z2, qm)
        z = z + rk['dz']
        w = w + rk['dw']
        t = t + dt
        if abs(rk['dz']) / abs(z) > 0.02:
            dt = dt / 2
        if abs(rk['dz']) / abs(z) < 0.005:
            dt = 2 * dt
        wi = w + 1j * z
        rold = r
        r = abs(z - z1)
        vk = 1.0 / sqrt(r * (1 + qm))
        no = np.conj(z - z1) / r
        wk = -vk * no * 1j
        wk = wk + wm1
        wout.append(wi)
        wkout.append(np.conj(wk))
    wout = np.array(wout)
    wkout = np.array(wkout)
    return {'vxi': wout.real, 'vyi': wout.imag, 'vkxi': wkout.real, 'vkyi': wkout.imag}


def vscale(porb, m1, q, vfs):
    # porb in days
    porb_sec = porb * 24 * 3600
    omega = 2 * pi / porb_sec
    a = (gg * m1 * msun * (1 + q)) ** (1.0 / 3.0) / omega ** (2.0 / 3.0)
    vs = omega * a / vfs
    return vs

# Drawing helpers expect a matplotlib axes available in calling code

def draw_2ndlobe(ax, q, incl, porb, m1=0.6, vfs=1e5, **plot_kwargs):
    rl1 = rlq1(q)
    l = lobes(q, rl1)
    cm = q / (1 + q)
    vs = vscale(porb, m1, q, vfs)
    si = sin(pi * incl / 180.0)
    xl = -l['y'] * si * vs
    yl = (l['x'] - cm) * si * vs
    ax.plot(xl, yl, **plot_kwargs)


def draw_stream(ax, q, incl, porb, m1=0.6, n=250, vfs=1e5, **plot_kwargs):
    rl1 = rlq1(q)
    st = stream(q, rl1, nmax=n)
    vs = vscale(porb, m1, q, vfs)
    si = sin(pi * incl / 180.0)
    ax.plot(st['vxi'] * si * vs, st['vyi'] * si * vs, **plot_kwargs)


def draw_kepler(ax, q, incl, porb, m1=0.6, n=250, vfs=1e5, **plot_kwargs):
    rl1 = rlq1(q)
    st = stream(q, rl1, nmax=n)
    vs = vscale(porb, m1, q, vfs)
    si = sin(pi * incl / 180.0)
    ax.plot(st['vkxi'] * si * vs, st['vkyi'] * si * vs, **plot_kwargs)


def draw_circle(ax, r, x0, y0, **plot_kwargs):
    theta = np.linspace(0, 2 * pi, 360)
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0
    ax.plot(x, y, **plot_kwargs)


def draw_center(ax, size=1, **plot_kwargs):
    ax.plot(0, 0, marker='+', markersize=8 * size, **plot_kwargs)


def draw_m2(ax, q, incl, porb, m1=0.6, vfs=1e5, size=1, **plot_kwargs):
    cm = q / (1 + q)
    vs = vscale(porb, m1, q, vfs)
    si = sin(pi * incl / 180.0)
    ax.plot(0, (1 - cm) * vs * si, marker='x', markersize=6 * size, **plot_kwargs)
    print(f"M2: Vy={(1-cm)*vs*si:.2f}")


def draw_m1(ax, q, incl, porb, m1=0.6, vfs=1e5, size=1, **plot_kwargs):
    cm = q / (1 + q)
    vs = vscale(porb, m1, q, vfs)
    si = sin(pi * incl / 180.0)
    ax.plot(0, -cm * vs * si, marker='x', markersize=6 * size, **plot_kwargs)
    print(f"M1: Vy={-cm*vs*si:.2f}")
