"""
simulate_math_applied_01.py
─────────────────────────────────────────────────────────────────────────────
EET Applied Mathematics — Probability, Statistics, Trigonometry, Geometry
Demonstrates job-qualification math concepts through real EET bench scenarios.

Plot 1 — Measurement Uncertainty (Probability / Normal Distribution)
  DMM reads TP3 rail 50 times. Shows Gaussian spread, 3-sigma limits,
  and how measurement uncertainty affects pass/fail decisions.
  Ties to: daily_workflow_simulation.html → 10:00 AM DMM sweep

Plot 2 — Test Yield & Process Capability (Statistical Inference / Cpk)
  Models a board population with tolerance-distributed component values.
  Calculates Cpk — the manufacturing quality index.
  Ties to: daily_workflow_simulation.html → MES log, NCR analysis

Plot 3 — AC Phasor & Trig (Trigonometry)
  RC circuit: phase angle φ = arctan(Xc/R), impedance |Z| = sqrt(R²+Xc²)
  Draws the phasor triangle and time-domain voltage/current waveforms.
  Ties to: fe_theory_to_bench_practice.html → Q1, Q2

Plot 4 — PCB Trace Geometry (Plane Geometry)
  Calculates trace length for an angled route between two pads.
  Models signal travel time and trace inductance from geometry.
  Ties to: signal_analysis_deep.py → RLC Bode (L_trace from trace length)

Run:
  python3 simulate_math_applied_01.py

Output → analysis_output/
  math_01_measurement_uncertainty.png
  math_02_process_capability.png
  math_03_phasor_trig.png
  math_04_trace_geometry.png
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os

np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "analysis_output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  'white',
    'axes.facecolor':    '#f8f9fa',
    'axes.edgecolor':    '#aaaaaa',
    'grid.color':        '#dddddd',
    'text.color':        '#222222',
})

NOMINAL_3V3     = 3.300   # V
SPEC_MIN_volts  = 3.135   # ±5% lower limit
SPEC_MAX_volts  = 3.465   # ±5% upper limit
BROWNOUT_volts  = 2.970   # MCU brownout threshold


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Measurement Uncertainty (Probability / Normal Distribution)
# ═══════════════════════════════════════════════════════════════════════════

def plot_measurement_uncertainty():
    """
    A DMM reading is never a single exact value — it has measurement uncertainty.
    Model 50 repeated readings of TP3 as a normal distribution.
    Calculate mean, standard deviation, 3-sigma limits, and probability of
    a good board being falsely rejected due to measurement noise alone.

    Key formula:
      68% of readings fall within  µ ± 1σ
      95% of readings fall within  µ ± 2σ
      99.7% of readings fall within µ ± 3σ
    """
    print("\n[1/4] Measurement Uncertainty — TP3 Rail (Probability / Normal Distribution)")

    mu_volts    = 3.285       # true rail voltage (slightly low but in spec)
    sigma_volts = 0.012       # DMM repeatability + actual rail noise: ~12mV 1-sigma

    n_readings  = 50
    readings_v  = np.random.normal(mu_volts, sigma_volts, n_readings)

    # Continuous PDF curve
    x_v   = np.linspace(mu_volts - 4*sigma_volts, mu_volts + 4*sigma_volts, 400)
    pdf   = (1 / (sigma_volts * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_v - mu_volts)/sigma_volts)**2)

    # Probability of reading falling outside spec (false reject)
    from scipy import stats
    p_below_spec = stats.norm.cdf(SPEC_MIN_volts, loc=mu_volts, scale=sigma_volts) * 100
    p_above_spec = (1 - stats.norm.cdf(SPEC_MAX_volts, loc=mu_volts, scale=sigma_volts)) * 100
    p_false_reject = p_below_spec + p_above_spec

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram + PDF overlay
    ax = axes[0]
    ax.hist(readings_v, bins=14, density=True, color='#1565c0', alpha=0.6,
            edgecolor='white', linewidth=0.5, label=f'{n_readings} DMM readings')
    ax.plot(x_v, pdf, color='#c62828', linewidth=2.0, label='Normal PDF')

    # Spec band shading
    ax.axvline(SPEC_MIN_volts, color='#e65100', linestyle='--', linewidth=1.5,
               label=f'Spec min {SPEC_MIN_volts}V')
    ax.axvline(SPEC_MAX_volts, color='#2e7d32', linestyle='--', linewidth=1.5,
               label=f'Spec max {SPEC_MAX_volts}V')
    ax.axvline(mu_volts,       color='#555555', linestyle=':',  linewidth=1.0,
               label=f'Mean {mu_volts:.3f}V')

    # 1σ / 2σ / 3σ fill
    for n_sig, alpha, lbl in [(1, 0.10, '±1σ  68%'),
                               (2, 0.07, '±2σ  95%'),
                               (3, 0.04, '±3σ  99.7%')]:
        ax.axvspan(mu_volts - n_sig*sigma_volts,
                   mu_volts + n_sig*sigma_volts,
                   alpha=alpha, color='#1565c0', label=lbl)

    ax.set_xlabel("TP3 Voltage Reading (V)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(f"TP3 Measurement Distribution\n"
                 f"µ = {mu_volts:.3f}V  ·  σ = {sigma_volts*1000:.0f}mV  ·  "
                 f"n = {n_readings} readings", fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.5)

    # Right: control chart (run chart of readings)
    ax2 = axes[1]
    ax2.plot(range(1, n_readings+1), readings_v, 'o-', color='#1565c0',
             linewidth=1.2, markersize=4, alpha=0.8, label='Individual readings')
    ax2.axhline(mu_volts,                    color='#555555', linestyle=':',  linewidth=1.0, label='Mean')
    ax2.axhline(mu_volts + 3*sigma_volts,    color='#c62828', linestyle='--', linewidth=1.2, label='+3σ UCL')
    ax2.axhline(mu_volts - 3*sigma_volts,    color='#c62828', linestyle='--', linewidth=1.2, label='−3σ LCL')
    ax2.axhline(SPEC_MIN_volts,              color='#e65100', linestyle='-',  linewidth=1.5, label='Spec limits')
    ax2.axhline(SPEC_MAX_volts,              color='#e65100', linestyle='-',  linewidth=1.5)
    ax2.axhline(BROWNOUT_volts,              color='#b71c1c', linestyle='-',  linewidth=1.5, label='Brownout threshold')

    ax2.set_xlabel("Reading Number", fontsize=11)
    ax2.set_ylabel("Voltage (V)", fontsize=11)
    ax2.set_title(f"Control Chart — TP3 Run Sequence\n"
                  f"False reject risk: {p_false_reject:.3f}%  "
                  f"({p_below_spec:.4f}% below spec)", fontsize=11)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylim(2.85, 3.55)
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUT, "math_01_measurement_uncertainty.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    print(f"  Mean:           µ = {np.mean(readings_v):.4f} V")
    print(f"  Std deviation:  σ = {np.std(readings_v)*1000:.2f} mV")
    print(f"  3σ UCL:         {mu_volts + 3*sigma_volts:.4f} V  (spec max: {SPEC_MAX_volts} V)")
    print(f"  3σ LCL:         {mu_volts - 3*sigma_volts:.4f} V  (spec min: {SPEC_MIN_volts} V)")
    print(f"  False reject probability: {p_false_reject:.4f}%")
    print(f"  Interpretation: board is GOOD but {p_false_reject:.4f}% of measurements "
          f"may read out-of-spec due to noise alone")
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Test Yield & Process Capability (Statistical Inference / Cpk)
# ═══════════════════════════════════════════════════════════════════════════

def plot_process_capability():
    """
    Cpk (Process Capability Index) measures how well a manufacturing process
    keeps output within spec limits. Higher Cpk = fewer defects.

    Cpk = min( (USL - µ) / 3σ ,  (µ - LSL) / 3σ )

    Cpk >= 1.33  → capable process  (industry standard target)
    Cpk  = 1.00  → 0.27% defect rate (2700 PPM)
    Cpk  < 1.00  → process not capable — too many out-of-spec units

    Models C2 ESR distribution across a batch of boards.
    """
    print("\n[2/4] Process Capability (Cpk) — C2 ESR Batch Analysis")

    esr_nominal_ohms = 0.050    # target ESR for new MLCC cap
    esr_spec_max     = 0.100    # spec limit (from instrument_automation.py)
    esr_spec_min     = 0.001    # physical lower bound (essentially 0)

    # Three scenarios: tight process, marginal process, out-of-control process
    scenarios = [
        ("Tight process\n(new MLCC lot)",     0.045, 0.008, '#2e7d32'),
        ("Marginal process\n(mixed lot)",      0.065, 0.018, '#e65100'),
        ("Out of control\n(aged/mixed stock)", 0.075, 0.030, '#c62828'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, (label, mu, sigma, color) in zip(axes, scenarios):
        x = np.linspace(0, 0.18, 600)
        pdf = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)

        # Cpk calculation
        cpu = (esr_spec_max - mu) / (3 * sigma)
        cpl = (mu - esr_spec_min) / (3 * sigma)
        cpk = min(cpu, cpl)

        # Defect rate (area outside spec)
        from scipy import stats
        p_defect = (1 - stats.norm.cdf(esr_spec_max, mu, sigma)) * 100

        ax.plot(x, pdf, color=color, linewidth=2.2)
        ax.fill_between(x, pdf, where=(x <= esr_spec_max),
                        color=color, alpha=0.20, label='Within spec')
        ax.fill_between(x, pdf, where=(x > esr_spec_max),
                        color='#c62828', alpha=0.50, label='Out of spec')

        ax.axvline(esr_spec_max, color='#c62828', linestyle='--', linewidth=1.5,
                   label=f'Spec max {esr_spec_max}Ω')
        ax.axvline(mu, color=color, linestyle=':', linewidth=1.2,
                   label=f'µ = {mu:.3f}Ω')

        cpk_color = '#2e7d32' if cpk >= 1.33 else ('#e65100' if cpk >= 1.0 else '#c62828')
        ax.set_title(f"{label}\nCpk = {cpk:.2f}  ·  Defect rate: {p_defect:.2f}%",
                     fontsize=10, color=cpk_color, fontweight='bold')
        ax.set_xlabel("C2 ESR (Ω)", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.5)
        ax.set_xlim(0, 0.18)

        print(f"  {label.replace(chr(10),' ')}: µ={mu:.3f}Ω  σ={sigma:.3f}Ω  "
              f"Cpk={cpk:.2f}  defects={p_defect:.2f}%")

    plt.suptitle("C2 ESR Process Capability Analysis — Cpk Across Three Scenarios\n"
                 "Cpk ≥ 1.33 = capable  ·  Cpk < 1.0 = out of control",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT, "math_02_process_capability.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — AC Phasor & Trigonometry
# ═══════════════════════════════════════════════════════════════════════════

def plot_phasor_trig():
    """
    RC circuit phasor analysis — the triangle that connects R, Xc, and Z.

    Key trig relationships:
      Xc  = 1 / (2π·f·C)             (capacitive reactance)
      |Z| = sqrt(R² + Xc²)           (impedance magnitude — Pythagorean theorem)
      φ   = arctan(Xc / R)           (phase angle — trig)
      V_R = V_s · cos(φ)             (resistor voltage — trig)
      V_C = V_s · sin(φ)             (capacitor voltage — trig)

    This is the same math behind the oscilloscope phase measurement between
    two channels: phase difference = arctan(time_offset / period × 2π)
    """
    print("\n[3/4] AC Phasor Analysis — RC Circuit (Trigonometry)")

    # RC circuit values matching Scenario 01 bypass cap analysis
    R_ohms   = 50.0      # source/trace resistance
    C_farad  = 22e-6     # bypass cap
    f_hz     = 1e3       # 1kHz ripple frequency (from FFT analysis)
    V_s      = 1.0       # normalized source voltage

    Xc_ohms  = 1 / (2 * np.pi * f_hz * C_farad)
    Z_ohms   = np.sqrt(R_ohms**2 + Xc_ohms**2)
    phi_rad  = np.arctan(Xc_ohms / R_ohms)
    phi_deg  = np.degrees(phi_rad)

    V_R      = V_s * np.cos(phi_rad)
    V_C      = V_s * np.sin(phi_rad)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # ── Left: impedance triangle (geometry + trig) ────────────────────────
    ax = axes[0]
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')

    R_norm  = R_ohms  / Z_ohms
    Xc_norm = Xc_ohms / Z_ohms

    # Triangle sides
    ax.annotate('', xy=(R_norm, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=2.5))
    ax.annotate('', xy=(R_norm, Xc_norm), xytext=(R_norm, 0),
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=2.5))
    ax.annotate('', xy=(R_norm, Xc_norm), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2.5))

    ax.text(R_norm/2, -0.08, f'R = {R_ohms:.0f}Ω', ha='center',
            fontsize=10, color='#1565c0', fontweight='bold')
    ax.text(R_norm + 0.06, Xc_norm/2, f'Xc = {Xc_ohms:.1f}Ω', ha='left',
            fontsize=10, color='#c62828', fontweight='bold')
    ax.text(R_norm/2 - 0.12, Xc_norm/2 + 0.05, f'|Z| = {Z_ohms:.1f}Ω', ha='center',
            fontsize=10, color='#2e7d32', fontweight='bold')

    # Phase angle arc
    theta_arc = np.linspace(0, phi_rad, 60)
    ax.plot(0.18*np.cos(theta_arc), 0.18*np.sin(theta_arc), color='#555555', lw=1.5)
    ax.text(0.22, 0.07, f'φ = {phi_deg:.1f}°\narctan(Xc/R)', fontsize=9, color='#555555')

    ax.set_title(f"Impedance Triangle\nPythagorean theorem + arctan\n"
                 f"f = {f_hz/1e3:.0f}kHz  ·  C = {C_farad*1e6:.0f}µF", fontsize=10)
    ax.set_xlabel("Resistance (normalized)", fontsize=10)
    ax.set_ylabel("Reactance (normalized)", fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_facecolor('#f8f9fa')

    # ── Center: time-domain V_source, V_R, V_C ────────────────────────────
    ax2  = axes[1]
    t_ms = np.linspace(0, 3/f_hz, 1000)
    omega = 2 * np.pi * f_hz

    v_source = V_s      * np.sin(omega * t_ms)
    v_r      = V_R      * np.sin(omega * t_ms)
    v_c      = V_C      * np.sin(omega * t_ms - np.pi/2)   # cap lags 90°

    ax2.plot(t_ms*1000, v_source, color='#555555', linewidth=1.8, linestyle='--',
             label='V_source')
    ax2.plot(t_ms*1000, v_r,      color='#1565c0', linewidth=2.0,
             label=f'V_R = {V_R:.3f}V  (cos φ)')
    ax2.plot(t_ms*1000, v_c,      color='#c62828', linewidth=2.0,
             label=f'V_C = {V_C:.3f}V  (sin φ)  lags 90°')

    ax2.set_xlabel("Time (ms)", fontsize=10)
    ax2.set_ylabel("Voltage (V normalized)", fontsize=10)
    ax2.set_title(f"Time Domain — V_R in phase, V_C lags 90°\n"
                  f"V_R = Vs·cos(φ)  ·  V_C = Vs·sin(φ)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.5)
    ax2.axhline(0, color='#aaaaaa', linewidth=0.8)

    # ── Right: frequency sweep — phase angle vs frequency ─────────────────
    ax3     = axes[2]
    f_sweep = np.logspace(1, 6, 500)
    Xc_sw   = 1 / (2 * np.pi * f_sweep * C_farad)
    phi_sw  = np.degrees(np.arctan(Xc_sw / R_ohms))
    Z_sw    = np.sqrt(R_ohms**2 + Xc_sw**2)

    ax3.semilogx(f_sweep, phi_sw, color='#7c3aed', linewidth=2.0,
                 label='Phase φ = arctan(Xc/R)')
    ax3.axvline(f_hz, color='#c62828', linestyle='--', linewidth=1.3,
                label=f'f = {f_hz/1e3:.0f}kHz  (φ={phi_deg:.1f}°)')
    ax3.axhline(45, color='#2e7d32', linestyle=':', linewidth=1.0,
                label='45° (R = Xc, resonant balance)')

    ax3.set_xlabel("Frequency (Hz)", fontsize=10)
    ax3.set_ylabel("Phase Angle φ (°)", fontsize=10)
    ax3.set_title(f"Phase Angle vs Frequency\nRC Low-Pass Filter  R={R_ohms}Ω  C={C_farad*1e6:.0f}µF",
                  fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.5, which='both')
    ax3.set_ylim(0, 95)

    plt.tight_layout()
    out = os.path.join(OUT, "math_03_phasor_trig.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    print(f"  f        = {f_hz/1e3:.1f} kHz")
    print(f"  Xc       = 1/(2π·f·C) = {Xc_ohms:.4f} Ω")
    print(f"  |Z|      = √(R²+Xc²)  = {Z_ohms:.4f} Ω   (Pythagorean theorem)")
    print(f"  φ        = arctan(Xc/R) = {phi_deg:.2f}°   (trig)")
    print(f"  V_R      = Vs·cos(φ)   = {V_R:.4f} V")
    print(f"  V_C      = Vs·sin(φ)   = {V_C:.4f} V")
    print(f"  Check:   V_R² + V_C²   = {V_R**2 + V_C**2:.6f}  (should = 1.0 — Pythagorean)")
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 — PCB Trace Geometry (Plane Geometry)
# ═══════════════════════════════════════════════════════════════════════════

def plot_trace_geometry():
    """
    PCB traces route at 45° angles between pads.
    Calculate exact trace lengths using the distance formula (plane geometry),
    then derive trace inductance and signal travel time from the geometry.

    Key geometry:
      Straight segment length: L = sqrt(Δx² + Δy²)
      45° segment:             L_45 = |Δx| · sqrt(2)   (when Δx = Δy)
      Total trace length:      sum of all segments
      Trace inductance:        ~1nH per mm
      Signal travel time:      t = L / v_prop,  v_prop ≈ 0.6c for FR4
    """
    print("\n[4/4] PCB Trace Geometry — Trace Length, Inductance, Travel Time")

    # Define a simple route: pad A → 45° jog → pad B
    # Coordinates in mm on a PCB
    pad_A   = np.array([5.0,  5.0])
    jog_1   = np.array([10.0, 5.0])    # horizontal segment
    jog_2   = np.array([14.0, 9.0])    # 45° diagonal segment
    pad_B   = np.array([20.0, 9.0])    # horizontal to destination

    waypoints = [pad_A, jog_1, jog_2, pad_B]
    labels    = ['Pad A\n(U4.PIN22)', 'Jog 1', 'Jog 2', 'Pad B\n(U6.PIN3)']

    # Segment lengths
    segs_mm = []
    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1][0] - waypoints[i][0]
        dy = waypoints[i+1][1] - waypoints[i][1]
        seg_len = np.sqrt(dx**2 + dy**2)   # distance formula
        seg_ang = np.degrees(np.arctan2(dy, dx))
        segs_mm.append((seg_len, seg_ang))
        print(f"  Segment {i+1}: Δx={dx:.1f}mm  Δy={dy:.1f}mm  "
              f"length=√({dx:.0f}²+{dy:.0f}²) = {seg_len:.3f}mm  angle={seg_ang:.1f}°")

    total_len_mm  = sum(s[0] for s in segs_mm)
    L_inductance  = total_len_mm * 1e-9    # 1nH per mm rule
    v_prop        = 0.6 * 3e8             # propagation velocity in FR4 (60% speed of light)
    t_prop_ps     = (total_len_mm * 1e-3) / v_prop * 1e12  # picoseconds

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: PCB route diagram ───────────────────────────────────────────
    ax = axes[0]
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]

    ax.plot(xs, ys, 'o-', color='#c8a000', linewidth=4, markersize=10,
            markerfacecolor='#ffcc00', markeredgecolor='#996600', zorder=5,
            label='Cu trace route')

    for i, (pt, lbl) in enumerate(zip(waypoints, labels)):
        ax.annotate(lbl, xy=pt, xytext=(pt[0], pt[1] + 0.8),
                    ha='center', fontsize=8.5, color='#333333')

    # Annotate each segment
    for i in range(len(waypoints)-1):
        mid = (waypoints[i] + waypoints[i+1]) / 2
        seg_len, seg_ang = segs_mm[i]
        ax.text(mid[0], mid[1] - 0.7, f'{seg_len:.2f}mm\n{seg_ang:.0f}°',
                ha='center', fontsize=8, color='#1565c0')

    # Dimension arrows
    ax.annotate('', xy=(pad_B[0], 3.5), xytext=(pad_A[0], 3.5),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
    ax.text((pad_A[0]+pad_B[0])/2, 3.0,
            f'Total trace = {total_len_mm:.2f} mm', ha='center', fontsize=9, color='#555555')

    ax.set_xlim(2, 23)
    ax.set_ylim(2, 12)
    ax.set_xlabel("X position (mm)", fontsize=10)
    ax.set_ylabel("Y position (mm)", fontsize=10)
    ax.set_title("PCB Trace Route — ADDR_BUS[4]\nU4.PIN22 → U6.PIN3  (from Scenario 01 boundary-scan fault)",
                 fontsize=10)
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, alpha=0.2, color='#444466')
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_color('#555555')
    ax.tick_params(colors='#888888')
    ax.legend(fontsize=9)

    # ── Right: derived parameters bar chart ──────────────────────────────
    ax2 = axes[1]
    params      = ['Total length\n(mm)', 'Inductance\n(nH)', 'Travel time\n(ps)']
    values      = [total_len_mm, L_inductance * 1e9, t_prop_ps]
    bar_colors  = ['#1565c0', '#7c3aed', '#2e7d32']

    bars = ax2.bar(params, values, color=bar_colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(values)*0.02,
                 f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel("Value", fontsize=10)
    ax2.set_title(f"Derived Parameters from Trace Geometry\n"
                  f"Inductance = 1nH/mm rule  ·  v_prop = 0.6c (FR4)", fontsize=10)
    ax2.grid(True, axis='y', alpha=0.5)

    # Annotation: this inductance is what the RLC Bode plot assumed
    ax2.text(0.5, 0.15,
             f"This {L_inductance*1e9:.1f}nH inductance is the\n"
             f"L_trace input to signal_analysis_deep.py\nPlot 2 (RLC Bode)",
             transform=ax2.transAxes, ha='center', fontsize=8.5,
             color='#7c3aed',
             bbox=dict(boxstyle='round', facecolor='#f3e8ff', alpha=0.7))

    plt.tight_layout()
    out = os.path.join(OUT, "math_04_trace_geometry.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    print(f"  Total trace length: {total_len_mm:.3f} mm")
    print(f"  Trace inductance:   {L_inductance*1e9:.2f} nH  (1nH/mm rule)")
    print(f"  Signal travel time: {t_prop_ps:.2f} ps  (v_prop = 0.6c in FR4)")
    print(f"  This L_trace value feeds directly into signal_analysis_deep.py Plot 2 (RLC Bode)")
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  EET Applied Mathematics — Job Qualification Evidence")
    print("  Probability · Statistical Inference · Trig · Geometry")
    print("=" * 62)
    print()
    print("  Ties to:")
    print("    daily_workflow_simulation.html")
    print("    fe_theory_to_bench_practice.html")
    print("    signal_analysis_deep.py")
    print("    scenario_power_rail_01/")

    plot_measurement_uncertainty()
    plot_process_capability()
    plot_phasor_trig()
    plot_trace_geometry()

    print()
    print("=" * 62)
    print("  All plots complete.  Output: analysis_output/")
    print()
    print("    math_01_measurement_uncertainty.png")
    print("    math_02_process_capability.png")
    print("    math_03_phasor_trig.png")
    print("    math_04_trace_geometry.png")
    print()
    print("  Open scenario_math_applied_01.html to see each concept")
    print("  explained with its bench application context.")
    print("=" * 62)


if __name__ == "__main__":
    main()
