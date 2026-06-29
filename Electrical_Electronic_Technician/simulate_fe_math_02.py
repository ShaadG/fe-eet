"""
simulate_fe_math_02.py
FE Electrical & Computer Practice Exam — Math & Statistics Problems mapped to EET bench
Problems: Q1 (Law of Sines), Q12 (Median), Q13 (Normal Distribution), Q14 (Binomial)
Run: python simulate_fe_math_02.py
Output: analysis_output/math_02_*.png (4 plots)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import comb
import os

OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_law_of_sines_triangle():
    """FE Q1 — Trigonometry: Law of Sines, plane geometry triangle → connector misalignment"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1e1e2e')
    for ax in axes:
        ax.set_facecolor('#2b2b3b')

    angle_A_deg = 45.0
    angle_B_deg = 110.0
    angle_C_deg = 25.0
    side_AC_cm  = 10.0

    # Law of Sines: AB/sin(C) = AC/sin(B)
    AB = side_AC_cm * np.sin(np.radians(angle_C_deg)) / np.sin(np.radians(angle_B_deg))

    print("=" * 60)
    print("FE Q1 — Law of Sines (Trigonometry / Plane Geometry)")
    print(f"  Angle A={angle_A_deg}°  Angle B={angle_B_deg}°  Angle C={angle_C_deg}°")
    print(f"  Side AC = {side_AC_cm} cm  (given)")
    print(f"  AB = AC × sin(C)/sin(B) = {side_AC_cm} × sin(25°)/sin(110°)")
    print(f"  AB = {side_AC_cm} × {np.sin(np.radians(angle_C_deg)):.4f} / {np.sin(np.radians(angle_B_deg)):.4f}")
    print(f"  AB = {AB:.2f} cm  → Answer A (4.5 cm)")

    # Vertex positions
    A = np.array([0.0, 0.0])
    C = np.array([side_AC_cm, 0.0])
    B = np.array([AB * np.cos(np.radians(angle_A_deg)),
                  AB * np.sin(np.radians(angle_A_deg))])

    ax = axes[0]
    triangle = plt.Polygon([A, B, C], fill=False, edgecolor='#00d4ff', linewidth=2.5)
    ax.add_patch(triangle)

    for pt, lbl, off in [(A, 'A', (-0.5, -0.5)), (B, 'B', (-0.3, 0.3)), (C, 'C', (0.2, -0.5))]:
        ax.text(pt[0]+off[0], pt[1]+off[1], lbl, color='#00d4ff', fontsize=14, fontweight='bold')

    ax.text(A[0]+0.7, A[1]+0.15, '45°',  color='#ffd700', fontsize=11)
    ax.text(B[0]-1.1, B[1]-0.8, '110°', color='#ffd700', fontsize=11)
    ax.text(C[0]-1.5, C[1]+0.15, '25°', color='#ffd700', fontsize=11)

    mid_AC = (A + C) / 2
    ax.text(mid_AC[0], mid_AC[1]-0.6, 'AC = 10 cm', color='#aaaaff', fontsize=10, ha='center')
    mid_AB = (A + B) / 2
    ax.text(mid_AB[0]-0.9, mid_AB[1], 'AB = ?', color='#ff6b6b', fontsize=11, fontweight='bold')

    ax.text(0.5, 0.10,
            f'Law of Sines:   AB / sin(C) = AC / sin(B)\n'
            f'AB = 10 × sin(25°) / sin(110°)\n'
            f'AB = 10 × 0.4226 / 0.9397 = {AB:.2f} cm',
            transform=ax.transAxes, color='#00ff88', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#00ff88', alpha=0.9))

    ax.set_xlim(-1, 12);  ax.set_ylim(-1.5, 5.5)
    ax.set_aspect('equal')
    ax.set_title('FE Q1 — Law of Sines (Plane Geometry + Trigonometry)', color='#00d4ff', fontsize=12, pad=10)
    ax.tick_params(colors='#888888');  ax.spines[:].set_color('#444444')

    # EET application panel
    ax2 = axes[1]
    ax2.set_facecolor('#2b2b3b')
    ax2.set_xlim(0, 14);  ax2.set_ylim(0, 7)

    pcb = plt.Rectangle((1, 1), 12, 5, fill=True, facecolor='#1a3a1a', edgecolor='#00aa44', linewidth=2)
    ax2.add_patch(pcb)

    ax2.plot(2, 2, 'o', color='#ffd700', markersize=12, zorder=5)
    ax2.text(1.3, 1.3, 'FID1\n(Ref A)', color='#ffd700', fontsize=8)

    ax2.plot(12, 2, 's', color='#ff6b6b', markersize=12, zorder=5)
    ax2.text(11.2, 1.3, 'J1 Ctr\n(Ref C)', color='#ff6b6b', fontsize=8)

    ax2.annotate('', xy=(12, 2), xytext=(2, 2),
                 arrowprops=dict(arrowstyle='<->', color='#aaaaff', lw=2))
    ax2.text(7, 1.4, 'Meas. = 10.0 cm', color='#aaaaff', fontsize=9, ha='center')

    B_pcb = np.array([2 + AB * np.cos(np.radians(45)),
                      2 + AB * np.sin(np.radians(45))])
    ax2.plot([2, B_pcb[0]], [2, B_pcb[1]], '--', color='#ff9944', linewidth=2.5)
    ax2.plot(B_pcb[0], B_pcb[1], '^', color='#ff9944', markersize=12, zorder=5)
    ax2.text(B_pcb[0]+0.15, B_pcb[1]+0.15, f'Pin Boss\n(calc {AB:.1f} cm from A)', color='#ff9944', fontsize=8)
    ax2.text(2.7, 2.5, '45°', color='#ffd700', fontsize=10)

    ax2.text(0.5, 0.10,
             f'Altium footprint shows 4.1 cm  |  Calculated = {AB:.1f} cm\n'
             f'Mismatch confirmed footprint error → NCR filed',
             transform=ax2.transAxes, color='#00ff88', fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#00ff88', alpha=0.9))

    ax2.set_title('EET Application: Connector Pin-to-Fiducial Distance\n(Assembly drawing verification)',
                  color='#ff9944', fontsize=11, pad=8)
    ax2.set_aspect('equal');  ax2.tick_params(colors='#888888');  ax2.spines[:].set_color('#444444')

    plt.suptitle('Trigonometry & Plane Geometry — FE Practice Q1 → EET Bench',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'math_02_triangle_trig.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1e1e2e')
    plt.close()
    print(f"  Saved: {path}")


def plot_normal_distribution_screening():
    """FE Q13 — Normal distribution / statistical inference → component ESR acceptance screening"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1e1e2e')
    for ax in axes:
        ax.set_facecolor('#2b2b3b')

    mu_q13    = 8.0
    sigma_q13 = 2.5
    x_thresh  = 15.5
    z_val     = (x_thresh - mu_q13) / sigma_q13
    p_fail    = 1 - stats.norm.cdf(x_thresh, mu_q13, sigma_q13)

    print("\nFE Q13 — Normal Distribution / Statistical Inference")
    print(f"  mu={mu_q13}, sigma={sigma_q13}, threshold={x_thresh}")
    print(f"  z = ({x_thresh} - {mu_q13}) / {sigma_q13} = {z_val:.1f}")
    print(f"  P(X > {x_thresh}) = 1 - F({z_val:.0f}) = 1 - 0.9987 = {p_fail:.4f}")
    print(f"  Answer A (0.0013)")

    x_q13  = np.linspace(mu_q13 - 4*sigma_q13, mu_q13 + 4*sigma_q13, 500)
    pdf_q13 = stats.norm.pdf(x_q13, mu_q13, sigma_q13)

    ax = axes[0]
    ax.plot(x_q13, pdf_q13, color='#00d4ff', linewidth=2.5)
    ax.fill_between(x_q13, pdf_q13, where=(x_q13 > x_thresh),
                    color='#ff4444', alpha=0.65, label=f'P(X>{x_thresh}) = {p_fail:.4f}')
    ax.fill_between(x_q13, pdf_q13, where=(x_q13 <= x_thresh),
                    color='#00aaff', alpha=0.20, label=f'P(X≤{x_thresh}) = 0.9987')
    ax.axvline(x_thresh, color='#ff4444', linestyle='--', linewidth=2)
    ax.axvline(mu_q13,   color='#ffd700', linestyle=':',  linewidth=2, label=f'μ = {mu_q13}')
    ax.text(x_thresh+0.15, max(pdf_q13)*0.72,
            f'z = {z_val:.0f}σ\nThreshold\n{x_thresh} min', color='#ff6b6b', fontsize=9)
    ax.set_xlabel('Call duration (min)', color='#aaaaaa')
    ax.set_ylabel('Probability Density', color='#aaaaaa')
    ax.set_title('FE Q13 — Normal Distribution\nP(X > 15.5 min | μ=8, σ=2.5) = 0.0013', color='#00d4ff', fontsize=12)
    ax.legend(fontsize=8, facecolor='#1e1e2e', labelcolor='white')
    ax.tick_params(colors='#888888');  ax.spines[:].set_color('#444444')

    # EET: ESR component screening
    mu_esr    = 0.050
    sigma_esr = 0.015
    esr_limit = mu_esr + 3 * sigma_esr
    p_esr_fail = 1 - stats.norm.cdf(esr_limit, mu_esr, sigma_esr)

    print(f"\n  EET ESR Screening: mu={mu_esr}Ω, sigma={sigma_esr}Ω, 3σ limit={esr_limit:.3f}Ω")
    print(f"  P(ESR > {esr_limit:.3f}) = {p_esr_fail:.4f} ({p_esr_fail*100:.2f}%)")

    x_esr  = np.linspace(mu_esr - 4*sigma_esr, mu_esr + 4*sigma_esr, 500)
    pdf_esr = stats.norm.pdf(x_esr, mu_esr, sigma_esr)

    ax2 = axes[1]
    ax2.plot(x_esr*1000, pdf_esr/1000, color='#00d4ff', linewidth=2.5)
    ax2.fill_between(x_esr*1000, pdf_esr/1000, where=(x_esr > esr_limit),
                     color='#ff4444', alpha=0.65,
                     label=f'Fail zone ({p_esr_fail*100:.2f}%)')
    ax2.fill_between(x_esr*1000, pdf_esr/1000, where=(x_esr <= esr_limit),
                     color='#00aaff', alpha=0.20, label='Pass zone (99.87%)')
    ax2.axvline(esr_limit*1000, color='#ff4444', linestyle='--', linewidth=2,
                label=f'3σ = {esr_limit*1000:.0f} mΩ')
    ax2.axvline(mu_esr*1000,    color='#ffd700', linestyle=':',  linewidth=2,
                label=f'μ = {mu_esr*1000:.0f} mΩ')
    ax2.set_xlabel('ESR (mΩ)', color='#aaaaaa')
    ax2.set_ylabel('Probability Density', color='#aaaaaa')
    ax2.set_title('EET Application: Cap ESR Acceptance Screening\n3σ boundary → 0.13% fail rate',
                  color='#ff9944', fontsize=11)
    ax2.legend(fontsize=8, facecolor='#1e1e2e', labelcolor='white')
    ax2.tick_params(colors='#888888');  ax2.spines[:].set_color('#444444')

    plt.suptitle('Statistical Inference — FE Practice Q13 → EET Component Screening',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'math_02_normal_screening.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1e1e2e')
    plt.close()
    print(f"  Saved: {path}")


def plot_binomial_defect_probability():
    """FE Q14 — Binomial distribution → ICT defect rate modeling for intermittent solder joint"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1e1e2e')
    for ax in axes:
        ax.set_facecolor('#2b2b3b')

    n_q14 = 10
    p_q14 = 0.5
    k_vals = np.arange(0, n_q14 + 1)
    pmf_q14 = [stats.binom.pmf(k, n_q14, p_q14) for k in k_vals]
    k_hi  = 4
    p_k4  = stats.binom.pmf(k_hi, n_q14, p_q14)

    print(f"\nFE Q14 — Binomial Distribution")
    print(f"  n={n_q14}, p={p_q14}, k={k_hi}")
    print(f"  C(10,4) = {int(comb(10,4))}")
    print(f"  P(X=4) = {int(comb(10,4))} × (0.5)^10 = {p_k4:.4f}  → Answer B (≈ 0.2)")

    ax = axes[0]
    colors_q14 = ['#ff4444' if k == k_hi else '#4488ff' for k in k_vals]
    ax.bar(k_vals, pmf_q14, color=colors_q14, edgecolor='#888888', linewidth=0.5, alpha=0.85)
    ax.axhline(p_k4, color='#ff4444', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(k_hi, p_k4+0.006, f'k=4\nP={p_k4:.3f}', color='#ff4444',
            ha='center', fontsize=10, fontweight='bold')
    ax.text(0.60, 0.87,
            f'P(X=k) = C(n,k)·p^k·(1-p)^(n-k)\n'
            f'C(10,4) = {int(comb(10,4))}\n'
            f'P(X=4) = {int(comb(10,4))} × (0.5)^10\n'
            f'       = {p_k4:.4f}  ≈  0.2',
            transform=ax.transAxes, color='#00ff88', fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#00ff88', alpha=0.9))
    ax.set_xlabel('Number of Heads (k)', color='#aaaaaa')
    ax.set_ylabel('P(X = k)', color='#aaaaaa')
    ax.set_title(f'FE Q14 — Binomial Distribution\nn=10, p=0.5, P(X=4) ≈ 0.2', color='#00d4ff', fontsize=12)
    ax.tick_params(colors='#888888');  ax.spines[:].set_color('#444444')

    # EET ICT application
    ax2 = axes[1]
    colors_ict = ['#ff4444' if k == k_hi else '#44cc66' for k in k_vals]
    ax2.bar(k_vals, pmf_q14, color=colors_ict, edgecolor='#888888', linewidth=0.5, alpha=0.85)
    ax2.text(k_hi, pmf_q14[k_hi]+0.006,
             f'P({k_hi} fail)\n={pmf_q14[k_hi]:.3f}', color='#ff4444', ha='center', fontsize=9)
    ax2.text(0.02, 0.92,
             'Marginal cold joint on QFN-Pin12:\n'
             'p=0.5 → 20% chance exactly 4/10 fail\n'
             '→ Pattern not statistically conclusive\n'
             '→ Increase batch or consecutive runs\n'
             '→ Escalate to reflow profile review',
             transform=ax2.transAxes, color='#aaffaa', fontsize=8, va='top',
             bbox=dict(boxstyle='round', facecolor='#1a2a1a', edgecolor='#44cc66', alpha=0.9))
    ax2.set_xlabel('Number of Boards Failing on QFN-P12 (k)', color='#aaaaaa')
    ax2.set_ylabel('Probability', color='#aaaaaa')
    ax2.set_title('EET Application: ICT Defect Rate Modeling\nMarginal solder joint, 10-board batch',
                  color='#ff9944', fontsize=11)
    ax2.tick_params(colors='#888888');  ax2.spines[:].set_color('#444444')

    plt.suptitle('Probability — FE Practice Q14 → EET ICT Defect Modeling',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'math_02_binomial_ict.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1e1e2e')
    plt.close()
    print(f"  Saved: {path}")


def plot_mean_vs_median_rail():
    """FE Q12 — Mean vs Median → voltage rail QC, skew detection"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1e1e2e')
    for ax in axes:
        ax.set_facecolor('#2b2b3b')

    data_q12  = [11, 11, 11, 11, 12, 13, 13, 14]
    mean_q12  = np.mean(data_q12)
    med_q12   = np.median(data_q12)

    print(f"\nFE Q12 — Mean vs Median")
    print(f"  Data: {data_q12}")
    print(f"  Mean   = {mean_q12:.1f}  (given)")
    print(f"  Median = {med_q12:.1f}   (4th={data_q12[3]}, 5th={data_q12[4]}, avg={med_q12:.1f})")
    print(f"  Answer C (11.5)")
    print(f"  Right-skewed: mean > median → outlier on upper end")

    ax = axes[0]
    x_idx = np.arange(len(data_q12))
    ax.bar(x_idx, data_q12, color='#4488ff', edgecolor='#888888', linewidth=0.5, alpha=0.85)
    ax.axhline(mean_q12, color='#ff4444', linestyle='--', linewidth=2, label=f'Mean = {mean_q12:.1f}')
    ax.axhline(med_q12,  color='#ffd700', linestyle='-',  linewidth=2, label=f'Median = {med_q12:.1f}')
    ax.set_ylim(8, 16)
    ax.set_xlabel('Measurement Index', color='#aaaaaa')
    ax.set_ylabel('Value', color='#aaaaaa')
    ax.set_title(f'FE Q12 — Mean vs Median\nDataset: {data_q12}', color='#00d4ff', fontsize=12)
    ax.legend(fontsize=9, facecolor='#1e1e2e', labelcolor='white')
    ax.text(0.5, 0.10,
            f'Mean ({mean_q12:.1f}) > Median ({med_q12:.1f}) → right-skewed distribution\n'
            f'Outlier (14) is pulling the mean up',
            transform=ax.transAxes, color='#00ff88', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#00ff88', alpha=0.9))
    ax.tick_params(colors='#888888');  ax.spines[:].set_color('#444444')

    # EET: 3.3V rail QC
    nominal_mv   = 3300.0
    deviations   = np.array(data_q12, dtype=float)
    actual_mv    = nominal_mv + deviations
    mean_rail    = np.mean(actual_mv)
    med_rail     = np.median(actual_mv)
    usl_mv       = nominal_mv + 50
    lsl_mv       = nominal_mv - 50

    print(f"\n  EET Rail QC: nominal={nominal_mv}mV, USL={usl_mv}mV, LSL={lsl_mv}mV")
    print(f"  Actual readings: {actual_mv.tolist()}")
    print(f"  Mean rail = {mean_rail:.1f}mV   Median rail = {med_rail:.1f}mV")

    ax2 = axes[1]
    bar_colors = ['#ff6b6b' if v > usl_mv else '#4488ff' for v in actual_mv]
    ax2.bar(np.arange(len(actual_mv))+1, actual_mv, color=bar_colors, edgecolor='#888888',
            linewidth=0.5, alpha=0.85)
    ax2.axhline(mean_rail, color='#ff4444', linestyle='--', linewidth=2,
                label=f'Mean = {mean_rail:.1f} mV')
    ax2.axhline(med_rail,  color='#ffd700', linestyle='-',  linewidth=2,
                label=f'Median = {med_rail:.1f} mV')
    ax2.axhline(usl_mv, color='#ff0000', linestyle=':', linewidth=1.5, label=f'USL = {usl_mv:.0f} mV')
    ax2.axhline(lsl_mv, color='#00aa44', linestyle=':', linewidth=1.5, label=f'LSL = {lsl_mv:.0f} mV')
    ax2.set_ylim(3240, 3360)
    ax2.set_xlabel('PCBA Board #', color='#aaaaaa')
    ax2.set_ylabel('3.3V Rail (mV)', color='#aaaaaa')
    ax2.set_title('EET Application: 3.3V Rail QC — 8-Board Batch\nMean ≠ Median → Outlier investigation',
                  color='#ff9944', fontsize=11)
    ax2.legend(fontsize=8, facecolor='#1e1e2e', labelcolor='white')
    ax2.text(0.02, 0.22,
             'Board 8 reads +14 mV (outlier)\n'
             '→ Inductor DCR higher than spec\n'
             '→ Median (11.5) more robust than\n'
             '   mean (12) for small-batch QC',
             transform=ax2.transAxes, color='#ffddaa', fontsize=8, va='bottom',
             bbox=dict(boxstyle='round', facecolor='#2a1a0a', edgecolor='#ff9944', alpha=0.9))
    ax2.tick_params(colors='#888888');  ax2.spines[:].set_color('#444444')

    plt.suptitle('Statistical Inference — FE Practice Q12 → EET Rail Voltage QC',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'math_02_mean_median_rail.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1e1e2e')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("FE EC Practice Exam — Math/Stats → EET Bench Scenarios")
    print("=" * 60)
    plot_law_of_sines_triangle()
    plot_normal_distribution_screening()
    plot_binomial_defect_probability()
    plot_mean_vs_median_rail()
    print("\nAll 4 plots generated in analysis_output/")
    print("  math_02_triangle_trig.png    — FE Q1  Trigonometry")
    print("  math_02_normal_screening.png — FE Q13 Normal Distribution")
    print("  math_02_binomial_ict.png     — FE Q14 Binomial Probability")
    print("  math_02_mean_median_rail.png — FE Q12 Mean vs Median")
