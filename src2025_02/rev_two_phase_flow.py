#!/usr/bin/env python3
"""
Representative Elementary Volume (REV) for Upscaled Two-Phase Flow.

Reference: McClure et al., 2025, Petrophysics 66(1), 68-79. DOI:10.30632/PJV66N1-2025a5

Implements:
  - Energy dissipation rate computation (Eq. 7)
  - Relative permeability from energy budget (Eqs. 8-11)
  - Temporal REV identification via convergence analysis
  - Ergodicity testing (spatial vs temporal vs ensemble)
  - Pressure/saturation fluctuation analysis
  - SCAL experiment duration guidelines
"""
import numpy as np

def energy_dissipation(mean_velocity, pressure_gradient):
    """Viscous dissipation rate Φ ≈ ū·∇p  [Eq. 7]."""
    return abs(mean_velocity * pressure_gradient)

def kr_from_energy(q_phase, mu, k_abs, grad_p):
    """kr_i = q_i·μ_i / (k_abs·|∇p|)  [from Eqs. 8-11]."""
    return max(q_phase * mu / (k_abs * abs(grad_p) + 1e-30), 0)

def temporal_rev_analysis(time, signal, n_windows=30):
    """Find optimal averaging window where CV of running mean drops below 5%."""
    dt = np.mean(np.diff(time))
    ws = np.unique(np.logspace(0, np.log10(len(signal)//2), n_windows).astype(int))
    ws = ws[ws>0]
    cv = []
    for w in ws:
        if w >= len(signal): cv.append(cv[-1] if cv else 1); continue
        rm = np.convolve(signal, np.ones(w)/w, 'valid')
        cv.append(np.std(rm)/(abs(np.mean(rm))+1e-30))
    cv = np.array(cv)
    opt = ws[np.argmax(cv<0.05)] if np.any(cv<0.05) else ws[-1]
    return ws, cv, opt

def ergodicity_test(spatial_avgs, temporal_avg, tol=0.10):
    sm = np.mean(spatial_avgs)
    dev = abs(sm-temporal_avg)/(abs(temporal_avg)+1e-30)
    return dict(spatial_mean=sm, temporal_mean=temporal_avg, deviation=dev, ergodic=dev<tol)

def fluctuation_analysis(pressure, saturation, time):
    def acorr_time(sig, dt):
        s = sig - sig.mean(); n = len(s)
        ac = np.correlate(s, s, 'full')[n-1:]; ac /= ac[0]+1e-30
        zc = np.argmax(ac<0); return (zc if zc>0 else n-1)*dt
    dt = np.mean(np.diff(time))
    tp = acorr_time(pressure, dt); ts = acorr_time(saturation, dt)
    return dict(p_cv=np.std(pressure)/(abs(np.mean(pressure))+1e-30),
                s_cv=np.std(saturation)/(abs(np.mean(saturation))+1e-30),
                tau_p=tp, tau_s=ts, recommended_time=max(tp,ts)*10)

def scal_duration_guide(Ca, L=0.05, phi=0.20, k=100e-15):
    tc = phi*L**2/k
    if Ca<1e-6: m,s = 50,"low Ca — significant ganglia dynamics"
    elif Ca<1e-4: m,s = 20,"moderate Ca — moderate fluctuations"
    else: m,s = 5,"high Ca — viscous-dominated"
    return dict(Ca=Ca, t_char=tc, duration_s=tc*m, duration_hr=tc*m/3600, note=s)

if __name__ == "__main__":
    np.random.seed(42)
    t = np.linspace(0, 3600, 1000)
    p = 5000 + 200*np.sin(2*np.pi*t/120) + 100*np.random.randn(len(t))
    s = 0.45 + 0.02*np.sin(2*np.pi*t/120) + 0.01*np.random.randn(len(t))
    fl = fluctuation_analysis(p, s, t)
    ws, cv, opt = temporal_rev_analysis(t, p)
    g = scal_duration_guide(1e-5)
    print(f"REV Two-Phase Flow — p_cv={fl['p_cv']:.4f}, s_cv={fl['s_cv']:.4f}")
    print(f"Optimal window: {opt} pts, recommended: {g['duration_hr']:.1f} hr")
    print(f"Ergodicity: {ergodicity_test(p[::100], np.mean(p))['ergodic']}")
