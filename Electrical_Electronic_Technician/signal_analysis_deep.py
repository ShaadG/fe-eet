"""
signal_analysis_deep.py
─────────────────────────────────────────────────────────────────────────────
EET Signal Analysis — scipy.signal + numpy + matplotlib
Five analyses that tie directly into the EET study files already built.

Analysis 1 — ESR Comparison (ties to scenario_power_rail_01, Q1 RC transient)
  Good cap (ESR=0.05Ω) vs bad cap (ESR=2.0Ω): voltage sag depth and recovery

Analysis 2 — RLC Bode Plot (ties to fe_theory_to_bench_practice.html Q2)
  Impedance magnitude and phase vs frequency for the 3.3V rail decoupling network
  Multiple cap values showing how ω₀ shifts with capacitance selection

Analysis 3 — FFT of Noisy Power Rail (ties to waveform_visualization.html Waveform 2)
  FFT reveals dominant noise frequency; connects oscilloscope time-domain view
  to frequency-domain interpretation used in signal processing

Analysis 4 — Transfer Function Step Response (ties to fe_theory_to_bench_practice.html Q4)
  H(s) = 50/(s² + 2ζωₙs + ωₙ²) step response for three damping ratios
  Annotates overshoot, settling time, ringing frequency for each ζ

Analysis 5 — Triangle Wave Harmonic Spectrum (ties to fe_theory_to_bench_practice.html Triangle)
  Triangle vs square vs sine harmonic content
  Shows why triangle wave generates less EMI than a square wave at the same frequency

Run:
  python3 signal_analysis_deep.py

Output (analysis_output/):
  01_esr_comparison.png
  02_rlc_bode.png
  03_fft_noise.png
  04_step_response.png
  05_triangle_harmonics.png
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "analysis_output")
os.makedirs(OUT, exist_ok=True)

# Shared matplotlib style — matches FE exam analysis.py
plt.rcParams.update({
    'font.family':        'DejaVu Serif',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   'white',
    'axes.facecolor':     '#f8f9fa',
    'axes.edgecolor':     '#aaaaaa',
    'grid.color':         '#dddddd',
    'text.color':         '#222222',
})

BROWNOUT_V      = 2.97    # MCU brownout reset threshold (V)
NOMINAL_3V3     = 3.30    # Nominal rail voltage (V)
C2_CAP_F        = 22e-6   # Bypass capacitor value (F)
LOAD_CURRENT_A  = 0.150   # MCU active-cycle load current (A)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — ESR Comparison: Good vs Degraded Bypass Capacitor
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_esr_comparison():
    """
    Computes and plots the 3.3V rail transient response for a good MLCC bypass
    cap (low ESR) vs a degraded cap (high ESR) during a 150mA MCU load step.

    τ = ESR × C  →  voltage sag depth = I × ESR  →  recovery v(t) = V_final - ΔV·e^(-t/τ)

    Ties to:
      scenario_power_rail_01/scenario_power_rail_01.html  (root cause C2)
      fe_theory_to_bench_practice.html  Q1 (RC transient)
      daily_workflow_simulation.html    10:00 AM ESR card
    """
    print("\n[1/5] ESR Comparison — Good vs Degraded Bypass Cap")

    esr_cases = [
        ("Good MLCC (low ESR)",   0.05,  '#2e7d32', '-'),   # < 0.1Ω → healthy
        ("Degraded cap (C2 S/N003)", 2.00, '#c62828', '--'), # 2.0Ω → failing board
    ]

    t_us   = np.linspace(0, 600, 5000)    # 0 to 600 µs
    t_s    = t_us * 1e-6

    # Two MCU load events: at t=100µs and t=350µs
    load_events_us = [100, 350]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for esr_ohms, color, style in [(c[1], c[2], c[3]) for c in esr_cases]:
        tau_s  = esr_ohms * C2_CAP_F       # τ = ESR × C
        v_sag  = LOAD_CURRENT_A * esr_ohms # instantaneous voltage drop = I × R

        v_rail = np.full_like(t_s, NOMINAL_3V3 - 0.12)  # slight DC offset (rail low)

        for t_evt_us in load_events_us:
            t_evt_s = t_evt_us * 1e-6
            for i, t in enumerate(t_s):
                dt = t - t_evt_s
                if -2e-6 <= dt < 0:
                    v_rail[i] -= v_sag * (-dt / 2e-6)
                elif 0 <= dt:
                    v_rail[i] -= v_sag * np.exp(-dt / tau_s)

        label = f"ESR = {esr_ohms:.2f} Ω  (τ = {tau_s*1e6:.1f} µs)"
        axes[0].plot(t_us, v_rail, color=color, linestyle=style,
                     linewidth=2.0, label=label)

    axes[0].axhline(NOMINAL_3V3,  color='#555555',  linestyle=':',  linewidth=1.0,
                    label=f'{NOMINAL_3V3}V nominal')
    axes[0].axhline(BROWNOUT_V,   color='#e65100',  linestyle='--', linewidth=1.5,
                    label=f'{BROWNOUT_V}V brownout threshold')
    axes[0].fill_between(t_us, BROWNOUT_V - 0.2, BROWNOUT_V,
                         color='#e65100', alpha=0.08, label='Brownout violation zone')

    axes[0].set_xlabel("Time (µs)", fontsize=11)
    axes[0].set_ylabel("Rail Voltage (V)", fontsize=11)
    axes[0].set_title("3.3V Rail — Good vs Degraded C2\n"
                      f"Load step: {LOAD_CURRENT_A*1000:.0f} mA, "
                      f"C = {C2_CAP_F*1e6:.0f} µF", fontsize=11)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].set_ylim(BROWNOUT_V - 0.25, NOMINAL_3V3 + 0.15)
    axes[0].grid(True, alpha=0.5)

    # Right subplot: τ vs ESR sweep (shows design space)
    esr_sweep = np.logspace(-2, 1, 400)   # 10mΩ to 10Ω
    tau_sweep  = esr_sweep * C2_CAP_F * 1e6  # µs

    axes[1].semilogx(esr_sweep, tau_sweep, color='#1565c0', linewidth=2.2)
    axes[1].axvline(0.05,  color='#2e7d32', linestyle='--', linewidth=1.3,
                    label='Good MLCC (0.05 Ω)')
    axes[1].axvline(2.00,  color='#c62828', linestyle='--', linewidth=1.3,
                    label='C2 on S/N003 (2.00 Ω)')
    axes[1].axhline(2.0,   color='#e65100', linestyle=':', linewidth=1.0,
                    label='τ = 2 µs (brownout margin limit)')

    axes[1].set_xlabel("ESR (Ω) — log scale", fontsize=11)
    axes[1].set_ylabel("Time constant  τ = ESR × C  (µs)", fontsize=11)
    axes[1].set_title(f"τ vs ESR for C = {C2_CAP_F*1e6:.0f} µF\n"
                      "Higher ESR → slower recovery → brownout risk", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.5, which='both')

    plt.tight_layout()
    out = os.path.join(OUT, "01_esr_comparison.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    tau_good = 0.05  * C2_CAP_F * 1e6
    tau_bad  = 2.00  * C2_CAP_F * 1e6
    print(f"  Good cap:  ESR=0.05Ω → τ = {tau_good:.2f} µs  (sag recovers in {tau_good*5:.1f} µs)")
    print(f"  Bad  cap:  ESR=2.00Ω → τ = {tau_bad:.1f} µs  (sag recovers in {tau_bad*5:.0f} µs)")
    print(f"  Brownout margin exceeded: {LOAD_CURRENT_A*2.00:.3f}V drop on bad cap vs "
          f"{LOAD_CURRENT_A*0.05:.4f}V on good cap")
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — RLC Bode Plot (Decoupling Network Impedance)
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_rlc_bode():
    """
    Impedance magnitude |Z(jω)| and phase vs frequency for series RLC
    (trace inductance + bypass capacitor + ESR).

    Multiple cap values show how changing C shifts ω₀, allowing the designer
    to tune the minimum-impedance frequency to match the switching regulator.

    Ties to:
      fe_theory_to_bench_practice.html  Q2 (RLC resonance)
      FE exam linear_systems/analysis.py  Q2 plot
    """
    print("\n[2/5] RLC Bode Plot — Decoupling Network Impedance")

    # PCB trace inductance: ~1nH/mm for typical microstrip, 10mm trace = 10nH
    L_trace_H = 10e-9       # 10 nH parasitic trace inductance
    R_trace   = 0.05        # trace resistance + ESR (low for good cap)

    cap_cases = [
        (100e-9,   "100 nF",  '#c62828'),   # small cap — f₀ higher
        (1e-6,     "1 µF",    '#e65100'),
        (10e-6,    "10 µF",   '#1565c0'),   # medium
        (100e-6,   "100 µF",  '#2e7d32'),   # large cap — f₀ lower
    ]

    f = np.logspace(3, 9, 4000)   # 1kHz to 1GHz
    omega = 2 * np.pi * f

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for C_val, C_label, color in cap_cases:
        Z_jw = R_trace + 1j * omega * L_trace_H + 1 / (1j * omega * C_val)
        Z_mag = np.abs(Z_jw)
        Z_phase_deg = np.angle(Z_jw, deg=True)

        f0 = 1 / (2 * np.pi * np.sqrt(L_trace_H * C_val))
        Q  = (1 / R_trace) * np.sqrt(L_trace_H / C_val)

        ax_mag.loglog(f, Z_mag, color=color, linewidth=1.8,
                      label=f"C = {C_label}  (f₀ = {f0/1e6:.2f} MHz, Q = {Q:.1f})")
        ax_phase.semilogx(f, Z_phase_deg, color=color, linewidth=1.5)

    # Mark switching regulator frequencies common in PCB power supplies
    for f_sw, label in [(300e3, "300kHz\nreg."), (1e6, "1MHz\nreg."), (5e6, "5MHz\nreg.")]:
        ax_mag.axvline(f_sw, color='#888888', linestyle=':', linewidth=1.0, alpha=0.7)
        ax_mag.text(f_sw * 1.1, 1e-3, label, fontsize=7, color='#888888', va='bottom')

    ax_mag.set_ylabel("|Z(jω)| (Ω)", fontsize=11)
    ax_mag.set_title("Decoupling Network Impedance vs Frequency\n"
                     f"L_trace = {L_trace_H*1e9:.0f} nH  ·  R_ESR = {R_trace} Ω  ·  "
                     "Multiple cap values", fontsize=11)
    ax_mag.legend(fontsize=9, loc='upper right')
    ax_mag.grid(True, alpha=0.5, which='both')
    ax_mag.set_ylim(1e-4, 1e4)

    ax_phase.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_phase.set_ylabel("Phase (°)", fontsize=11)
    ax_phase.set_title("Phase Response", fontsize=10)
    ax_phase.axhline(0, color='#555555', linewidth=0.8, linestyle=':')
    ax_phase.grid(True, alpha=0.5, which='both')
    ax_phase.set_ylim(-95, 95)

    plt.tight_layout()
    out = os.path.join(OUT, "02_rlc_bode.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    for C_val, C_label, _ in cap_cases:
        f0 = 1 / (2 * np.pi * np.sqrt(L_trace_H * C_val))
        Q  = (1 / R_trace) * np.sqrt(L_trace_H / C_val)
        print(f"  C = {C_label:8s}  →  f₀ = {f0/1e6:.4f} MHz  (ω₀ = {2*np.pi*f0:.0f} rad/s),  Q = {Q:.1f}")
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — FFT of Noisy Power Rail
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_fft_noisy_rail():
    """
    Generates a simulated noisy 3.3V power rail signal, then computes its FFT.
    Shows how the frequency-domain view reveals dominant noise sources that are
    hard to identify from the oscilloscope time-domain display alone.

    Signal components:
      3.18V DC offset (slightly low, degraded cap)
      1kHz fundamental ripple (switching regulator noise)
      3kHz 3rd harmonic (nonlinear switching artifacts)
      Broadband random noise floor

    Ties to:
      waveform_visualization.html  Waveform 2 (noisy sine)
      fe_theory_to_bench_practice.html  Q3 (Laplace poles of sinusoidal components)
    """
    print("\n[3/5] FFT Analysis — Noisy 3.3V Power Rail")

    f_s    = 1e6          # sample rate: 1 MHz (realistic for USB scope)
    t_total = 0.01        # 10ms capture window
    N       = int(f_s * t_total)
    t       = np.arange(N) / f_s

    # Construct the signal
    v_dc        = 3.18                            # DC level (low — degraded cap)
    v_1khz      = 0.15 * np.sin(2*np.pi*1e3*t)   # 1kHz ripple: 150mV peak (regulator noise)
    v_3khz      = 0.04 * np.sin(2*np.pi*3e3*t)   # 3rd harmonic: 40mV peak
    v_50khz     = 0.02 * np.sin(2*np.pi*50e3*t)  # 50kHz transient artifact
    v_noise     = 0.015 * np.random.randn(N)     # broadband noise floor
    v_signal    = v_dc + v_1khz + v_3khz + v_50khz + v_noise

    # FFT
    V_fft     = np.fft.rfft(v_signal - v_dc)    # remove DC before FFT
    freqs     = np.fft.rfftfreq(N, d=1/f_s)
    V_mag_db  = 20 * np.log10(np.abs(V_fft) / (N / 2) + 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Time domain (oscilloscope view)
    t_ms   = t * 1000
    n_show = int(0.005 * f_s)   # show first 5ms (5 ripple cycles)
    axes[0].plot(t_ms[:n_show], v_signal[:n_show],
                 color='#ff7043', linewidth=0.9, alpha=0.85, label='TP3 — noisy rail')
    axes[0].axhline(NOMINAL_3V3, color='#2e7d32', linestyle='--', linewidth=1.2,
                    label=f'{NOMINAL_3V3}V nominal')
    axes[0].axhline(BROWNOUT_V,  color='#c62828', linestyle='--', linewidth=1.5,
                    label=f'{BROWNOUT_V}V brownout threshold')
    axes[0].set_xlabel("Time (ms)", fontsize=11)
    axes[0].set_ylabel("Voltage (V)", fontsize=11)
    axes[0].set_title("Time Domain — Oscilloscope View  (what you see on the bench)", fontsize=11)
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].set_ylim(2.9, 3.55)
    axes[0].grid(True, alpha=0.5)

    # Frequency domain (FFT — what the scope's FFT mode or spectrum analyzer shows)
    f_khz  = freqs / 1e3
    f_mask = f_khz <= 200   # plot up to 200kHz

    axes[1].plot(f_khz[f_mask], V_mag_db[f_mask],
                 color='#1565c0', linewidth=1.0, alpha=0.9)
    axes[1].fill_between(f_khz[f_mask], V_mag_db[f_mask], -120,
                         alpha=0.08, color='#1565c0')

    # Annotate dominant peaks
    for f_ann, label, yoff in [(1, "1kHz\n(fundamental ripple)", 8),
                                (3, "3kHz\n(3rd harmonic)", 8),
                                (50, "50kHz\n(transient)", 8)]:
        idx = np.argmin(np.abs(freqs - f_ann * 1e3))
        axes[1].annotate(label,
                         xy=(f_khz[idx], V_mag_db[idx]),
                         xytext=(f_khz[idx] + 5, V_mag_db[idx] + yoff),
                         fontsize=8.5, color='#c62828',
                         arrowprops=dict(arrowstyle='->', color='#c62828', lw=0.8))

    axes[1].set_xlabel("Frequency (kHz)", fontsize=11)
    axes[1].set_ylabel("Magnitude (dB)", fontsize=11)
    axes[1].set_title("Frequency Domain — FFT View  (reveals noise sources hidden in time-domain)", fontsize=11)
    axes[1].set_xlim(0, 200)
    axes[1].set_ylim(-90, 10)
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUT, "03_fft_noise.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    idx1k  = np.argmin(np.abs(freqs - 1e3))
    idx3k  = np.argmin(np.abs(freqs - 3e3))
    idx50k = np.argmin(np.abs(freqs - 50e3))
    print(f"  DC level:       {v_dc:.3f} V")
    print(f"  1kHz peak:      {V_mag_db[idx1k]:.1f} dB  (dominant — switching ripple)")
    print(f"  3kHz peak:      {V_mag_db[idx3k]:.1f} dB  (3rd harmonic of ripple)")
    print(f"  50kHz peak:     {V_mag_db[idx50k]:.1f} dB  (transient artifact)")
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Transfer Function Step Response (Damping Ratio Comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_step_response():
    """
    Step response of H(s) = ωₙ²/(s² + 2ζωₙs + ωₙ²) for three damping ratios.
    Uses scipy.signal.step for accurate computation.

    Annotates:
      - Overshoot percentage
      - Rise time (10% → 90%)
      - Settling time (±2% band)
      - Ringing frequency (underdamped only)

    Ties to:
      fe_theory_to_bench_practice.html  Q4 (transfer function, ζ=0.495)
      FE exam linear_systems/analysis.py  Q4 plot (same H(s))
      waveform_boundary_scan.html  Signal Waveforms tab
    """
    print("\n[4/5] Transfer Function Step Response — Damping Ratio Comparison")

    wn = np.sqrt(50)   # natural frequency from FE exam Q4: H(s) = 50/(s²+7s+50)

    # Three cases: highly underdamped, Q4 value, critically damped
    zeta_cases = [
        (0.15,  "ζ = 0.15 (highly underdamped)",    '#c62828'),
        (0.495, "ζ = 0.495 (FE exam Q4)",            '#1565c0'),
        (1.00,  "ζ = 1.00  (critically damped)",     '#2e7d32'),
    ]

    t = np.linspace(0, 8, 3000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for zeta, label, color in zeta_cases:
        # Build transfer function: H(s) = wn²/(s² + 2ζwn·s + wn²)
        num = [wn**2]
        den = [1, 2 * zeta * wn, wn**2]
        sys  = signal.TransferFunction(num, den)
        t_out, y_out = signal.step(sys, T=t)

        axes[0].plot(t_out, y_out, color=color, linewidth=2.0, label=label)

        # Annotate overshoot for underdamped cases
        if zeta < 1.0:
            os_pct = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100
            y_peak = np.max(y_out)
            t_peak = t_out[np.argmax(y_out)]
            axes[0].annotate(f"OS = {os_pct:.1f}%",
                             xy=(t_peak, y_peak),
                             xytext=(t_peak + 0.3, y_peak + 0.04),
                             fontsize=8, color=color,
                             arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    # ±2% settling band
    axes[0].fill_between(t, 0.98, 1.02, alpha=0.07, color='#555555', label='±2% settling band')
    axes[0].axhline(1.0, color='#555555', linewidth=0.8, linestyle=':', label='Final value')
    axes[0].set_xlabel("Time (normalized, s)", fontsize=11)
    axes[0].set_ylabel("Step Response y(t)", fontsize=11)
    axes[0].set_title(f"H(s) = ωₙ² / (s² + 2ζωₙs + ωₙ²)\n"
                      f"ωₙ = √50 ≈ {wn:.2f} rad/s  (from FE exam Q4)", fontsize=11)
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].set_ylim(-0.1, 1.6)
    axes[0].grid(True, alpha=0.5)

    # Right: pole-zero map for each ζ
    for zeta, label, color in zeta_cases:
        if zeta < 1.0:
            sigma = -zeta * wn
            wd    = wn * np.sqrt(1 - zeta**2)
            axes[1].plot(sigma,  wd,  'x', color=color, markersize=12,
                         markeredgewidth=2.2, label=label)
            axes[1].plot(sigma, -wd,  'x', color=color, markersize=12,
                         markeredgewidth=2.2)
        else:
            # Real repeated poles at -wn
            axes[1].plot(-wn, 0, 'o', color=color, markersize=10,
                         markeredgewidth=2.0, fillstyle='none', label=label)

    # ωₙ circle
    theta = np.linspace(0, 2*np.pi, 300)
    axes[1].plot(wn * np.cos(theta), wn * np.sin(theta),
                 color='#1565c0', linestyle='--', linewidth=0.8, alpha=0.4,
                 label=f'|s| = ωₙ ≈ {wn:.2f}')
    axes[1].axhline(0, color='#555555', linewidth=0.8)
    axes[1].axvline(0, color='#555555', linewidth=0.8, label='jω axis (stability boundary)')

    axes[1].set_xlabel("Real  σ", fontsize=11)
    axes[1].set_ylabel("Imaginary  jω", fontsize=11)
    axes[1].set_title("Pole-Zero Map\n(all poles in LHP → stable for all ζ > 0)", fontsize=11)
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].set_xlim(-12, 3)
    axes[1].set_ylim(-10, 10)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUT, "04_step_response.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    print(f"  ωₙ = √50 ≈ {wn:.4f} rad/s  (natural frequency from FE exam Q4)")
    for zeta, label, _ in zeta_cases:
        if zeta < 1.0:
            os_pct  = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100
            wd      = wn * np.sqrt(1 - zeta**2)
            t_ring  = 2 * np.pi / wd
            print(f"  ζ = {zeta:.3f}: overshoot = {os_pct:.1f}%,  "
                  f"ring period ≈ {t_ring:.3f} s  (scale to circuit frequency)")
        else:
            print(f"  ζ = {zeta:.2f}: critically damped — no overshoot, fastest non-ringing response")
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5 — Triangle Wave Harmonic Spectrum
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_triangle_harmonics():
    """
    Computes and plots the harmonic content of triangle, square, and sine waves.
    Shows why triangle generates less EMI: harmonic amplitudes fall as 1/n²
    compared to 1/n for square wave.

    Also shows Fourier series convergence: adding more harmonics to a triangle
    approximation yields closer agreement with the ideal waveform.

    Ties to:
      fe_theory_to_bench_practice.html  Triangle Wave section
      waveform_visualization.html        Waveform 3 (square wave)
    """
    print("\n[5/5] Triangle Wave Harmonic Spectrum")

    f0   = 1e3    # fundamental frequency: 1kHz
    A    = 1.0    # amplitude (normalized)
    N_max = 25    # number of harmonics to include

    t = np.linspace(0, 3/f0, 10000)  # 3 full cycles

    # Ideal triangle wave (piecewise linear)
    def triangle_ideal(t, f, A):
        T   = 1 / f
        p   = (t % T) / T           # phase 0–1
        return A * (4 * np.abs(p - 0.5) - 1)

    # Fourier series for triangle: v(t) = (8A/π²) Σ [(-1)^n / (2n+1)²] sin((2n+1)ωt)
    def triangle_fourier(t, f, A, n_terms):
        w0 = 2 * np.pi * f
        v  = np.zeros_like(t)
        for n in range(n_terms):
            k  = 2 * n + 1                          # odd harmonics only
            v += ((-1)**n / k**2) * np.sin(k * w0 * t)
        return v * (8 * A / np.pi**2)

    # Fourier series for square: v(t) = (4A/π) Σ [1/(2n+1)] sin((2n+1)ωt)
    def square_fourier(t, f, A, n_terms):
        w0 = 2 * np.pi * f
        v  = np.zeros_like(t)
        for n in range(n_terms):
            k  = 2 * n + 1
            v += (1 / k) * np.sin(k * w0 * t)
        return v * (4 * A / np.pi)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    # ── Top left: Fourier series convergence ─────────────────────────────────
    ax_conv = fig.add_subplot(gs[0, 0])
    v_ideal = triangle_ideal(t, f0, A)
    ax_conv.plot(t*1000, v_ideal, color='#555555', linewidth=2.5,
                 linestyle='--', label='Ideal triangle', zorder=5)
    for n_terms, color, alpha in [(1, '#c62828', 0.7), (3, '#e65100', 0.8),
                                   (7, '#1565c0', 0.9), (15, '#2e7d32', 1.0)]:
        v_approx = triangle_fourier(t, f0, A, n_terms)
        ax_conv.plot(t*1000, v_approx, color=color, linewidth=1.5, alpha=alpha,
                     label=f'N = {n_terms} harmonics')

    ax_conv.set_xlabel("Time (ms)", fontsize=10)
    ax_conv.set_ylabel("Amplitude (V)", fontsize=10)
    ax_conv.set_title("Fourier Series Convergence\n"
                      "Triangle wave reconstruction (more harmonics = better fit)", fontsize=10)
    ax_conv.legend(fontsize=8)
    ax_conv.grid(True, alpha=0.5)

    # ── Top right: Harmonic amplitude comparison ──────────────────────────────
    ax_harm = fig.add_subplot(gs[0, 1])
    harmonics = np.arange(1, 2*N_max, 2)   # odd harmonics: 1, 3, 5, ...

    amp_triangle = np.abs((8 * A / np.pi**2) * ((-1)**((harmonics-1)//2)) / harmonics**2)
    amp_square   = np.abs((4 * A / np.pi) / harmonics)
    amp_sine     = np.zeros_like(harmonics, dtype=float)
    amp_sine[0]  = A   # only fundamental

    x_pos = np.arange(len(harmonics))
    width = 0.25

    ax_harm.bar(x_pos - width, amp_sine,     width, color='#1565c0', alpha=0.8, label='Sine')
    ax_harm.bar(x_pos,         amp_triangle, width, color='#2e7d32', alpha=0.8, label='Triangle (1/n²)')
    ax_harm.bar(x_pos + width, amp_square,   width, color='#c62828', alpha=0.8, label='Square (1/n)')

    ax_harm.set_xticks(x_pos)
    ax_harm.set_xticklabels([f"{h}f₀" for h in harmonics], fontsize=8)
    ax_harm.set_xlabel("Harmonic", fontsize=10)
    ax_harm.set_ylabel("Amplitude (V)", fontsize=10)
    ax_harm.set_title("Harmonic Amplitude Comparison\n"
                      "Triangle 1/n² rolloff  vs  Square 1/n rolloff", fontsize=10)
    ax_harm.legend(fontsize=9)
    ax_harm.grid(True, axis='y', alpha=0.5)

    # ── Bottom left: dB amplitude vs harmonic number (EMI context) ───────────
    ax_db = fig.add_subplot(gs[1, 0])
    n_range = np.arange(1, 2*20, 2, dtype=float)
    amp_tri_db = 20 * np.log10(np.abs((8*A/np.pi**2) / n_range**2))
    amp_sq_db  = 20 * np.log10(np.abs((4*A/np.pi)   / n_range))

    ax_db.plot(n_range, amp_tri_db, 'o-', color='#2e7d32', linewidth=1.8,
               markersize=5, label='Triangle (-40dB/decade)')
    ax_db.plot(n_range, amp_sq_db,  's-', color='#c62828', linewidth=1.8,
               markersize=5, label='Square (-20dB/decade)')

    # Reference slope lines
    ax_db.plot(n_range, amp_tri_db[0] - 40*np.log10(n_range),
               color='#2e7d32', linestyle=':', linewidth=1.0, alpha=0.5, label='-40dB/decade ref')
    ax_db.plot(n_range, amp_sq_db[0]  - 20*np.log10(n_range),
               color='#c62828', linestyle=':', linewidth=1.0, alpha=0.5, label='-20dB/decade ref')

    ax_db.set_xlabel("Harmonic number n (odd)", fontsize=10)
    ax_db.set_ylabel("Amplitude (dB)", fontsize=10)
    ax_db.set_title("Harmonic Amplitude — dB Scale\n"
                    "Triangle rolls off 2× faster → less EMI at high frequencies", fontsize=10)
    ax_db.legend(fontsize=8)
    ax_db.grid(True, alpha=0.5)

    # ── Bottom right: waveform comparison (time domain) ───────────────────────
    ax_td = fig.add_subplot(gs[1, 1])
    t_2cyc = np.linspace(0, 2/f0, 4000)

    v_tri_td  = triangle_ideal(t_2cyc, f0, A)
    v_sq_td   = np.sign(np.sin(2 * np.pi * f0 * t_2cyc)) * A
    v_sin_td  = A * np.sin(2 * np.pi * f0 * t_2cyc)

    ax_td.plot(t_2cyc*1000, v_sin_td, color='#1565c0', linewidth=2.0,
               label='Sine (1 harmonic)', alpha=0.9)
    ax_td.plot(t_2cyc*1000, v_tri_td, color='#2e7d32', linewidth=2.0,
               label='Triangle (1/n²)')
    ax_td.plot(t_2cyc*1000, v_sq_td,  color='#c62828', linewidth=2.0,
               label='Square (1/n)',    linestyle='--')

    ax_td.set_xlabel("Time (ms)", fontsize=10)
    ax_td.set_ylabel("Amplitude (V)", fontsize=10)
    ax_td.set_title("Waveform Comparison — Same Frequency and Amplitude\n"
                    "Oscilloscope view — 2 cycles at 1kHz", fontsize=10)
    ax_td.legend(fontsize=9)
    ax_td.grid(True, alpha=0.5)

    plt.suptitle("Triangle Wave Harmonic Analysis\n"
                 "f₀ = 1kHz  ·  Fourier series  ·  EMI comparison",
                 fontsize=12, fontweight='bold', y=1.01)

    out = os.path.join(OUT, "05_triangle_harmonics.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    print(f"  Triangle  3rd harmonic amplitude: {amp_triangle[1]:.4f} V  "
          f"({20*np.log10(amp_triangle[1]/amp_triangle[0]):.1f} dB below fundamental)")
    print(f"  Square    3rd harmonic amplitude: {amp_square[1]:.4f} V  "
          f"({20*np.log10(amp_square[1]/amp_square[0]):.1f} dB below fundamental)")
    print(f"  Triangle rolls off at -40 dB/decade; square at -20 dB/decade")
    print(f"  At harmonic 9: triangle is {20*np.log10(amp_triangle[4]/amp_square[4]):.1f} dB "
          f"lower than square → significantly less radiated EMI")
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  EET Signal Analysis — scipy.signal + numpy + matplotlib")
    print("  Ties to: fe_theory_to_bench_practice.html")
    print("           scenario_power_rail_01/")
    print("           waveform_visualization.html")
    print("           daily_workflow_simulation.html")
    print("=" * 62)

    analysis_esr_comparison()
    analysis_rlc_bode()
    analysis_fft_noisy_rail()
    analysis_step_response()
    analysis_triangle_harmonics()

    print()
    print("=" * 62)
    print("  All analyses complete.  Output: analysis_output/")
    print()
    print("  Files generated:")
    print("    01_esr_comparison.png  → ESR vs brownout margin")
    print("    02_rlc_bode.png        → Decoupling network impedance")
    print("    03_fft_noise.png       → FFT: time-domain vs frequency-domain")
    print("    04_step_response.png   → Transfer fn step response (3 ζ values)")
    print("    05_triangle_harmonics.png → Harmonic spectrum comparison")
    print()
    print("  Open analysis_eet.html to see these tied back to EET workflow.")
    print("=" * 62)


if __name__ == "__main__":
    main()
