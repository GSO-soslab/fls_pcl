import numpy as np
import matplotlib.pyplot as plt


def plot_distributions():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    # ── Graph 1: Probability vs Intensity ─────────────────────────────────────
    lower    = 22
    upper    = 50
    min_prob = 0.2
    max_prob = 0.9

    intensities = np.arange(0, 256, dtype=np.float32)
    prob_intensity = np.full_like(intensities, min_prob)

    mid_mask = (intensities > lower) & (intensities < upper)
    prob_intensity[mid_mask] = min_prob + (
        (intensities[mid_mask] - lower) / (upper - lower)
    ) * (max_prob - min_prob)
    prob_intensity[intensities >= upper] = max_prob

    ax1.plot(intensities, prob_intensity, linewidth=2)
    ax1.axvline(x=lower, color='r', linestyle='--', alpha=0.6, label=f'Lower bound ({lower})')
    ax1.axvline(x=upper, color='g', linestyle='--', alpha=0.6, label=f'Upper bound ({upper})')
    ax1.axhline(y=min_prob, color='gray', linestyle=':',  alpha=0.5, label=f'Min prob ({min_prob})')
    ax1.axhline(y=max_prob, color='gray', linestyle='-.', alpha=0.5, label=f'Max prob ({max_prob})')
    ax1.set_xlabel('Intensity (0–255)', fontsize=9)
    ax1.set_ylabel('Probability', fontsize=9)
    ax1.set_title('Probability vs Intensity', fontsize=10)
    ax1.set_xlim([0, 255])
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ── Graph 2: Beam directivity (sinc) vs elevation angle ───────────────────
    c                  = 1500
    f                  = 1.2e6
    vertical_beamwidth = 12.0
    aperture_size      = 0.0067

    wavelength = c / f
    k          = 2 * np.pi / wavelength

    elevation_angles = np.arange(
        -vertical_beamwidth / 2,
        vertical_beamwidth / 2 + 1,
        1,
        dtype=np.float32,
    )
    angles_rad = np.deg2rad(elevation_angles)

    temp = (k * aperture_size / 2) * np.sin(angles_rad)
    prob_elevation = np.ones_like(temp, dtype=np.float32)
    non_zero = np.abs(temp) > 1e-10
    prob_elevation[non_zero] = np.sin(temp[non_zero]) / temp[non_zero]
    prob_elevation = np.array([round(x, 2) for x in prob_elevation])

    ax2.plot(elevation_angles, prob_elevation, 'b-o', linewidth=2, markersize=6)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Elevation Angle (degrees)', fontsize=9)
    ax2.set_ylabel('Beam Probability', fontsize=9)
    ax2.set_title('Beam Probability vs Elevation Angle', fontsize=10)
    ax2.set_ylim([0.4, 1.0])
    ax2.grid(True, alpha=0.3)

    # ── Graph 3: Joint probability (intensity × elevation) ────────────────────
    # Outer product: rows = elevation, cols = intensity
    joint = np.outer(prob_elevation, prob_intensity)  # (n_elev, n_intensity)

    im = ax3.imshow(
        joint,
        aspect='auto',
        origin='lower',
        extent=[intensities[0], intensities[-1], elevation_angles[0], elevation_angles[-1]],
        cmap='hot',
    )
    fig.colorbar(im, ax=ax3, label='Joint Probability')
    ax3.axvline(x=lower, color='cyan', linestyle='--', alpha=0.6, label=f'Intensity lower ({lower})')
    ax3.axvline(x=upper, color='lime',  linestyle='--', alpha=0.6, label=f'Intensity upper ({upper})')
    ax3.axhline(y=0,     color='white', linestyle=':',  alpha=0.5, label='Boresight (0°)')
    ax3.set_xlabel('Intensity (0–255)', fontsize=9)
    ax3.set_ylabel('Elevation Angle (degrees)', fontsize=9)
    ax3.set_title('Joint Probability (Intensity × Elevation)', fontsize=10)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_distributions()
