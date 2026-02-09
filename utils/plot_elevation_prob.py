import numpy as np
import matplotlib.pyplot as plt

def plot_directivity():
        c = 1500
        f = 1.2*1000*1000
        vertical_beamwidth = 12.0
        aperture_size = 0.0065
        elevation_beam_edge_probability = 0.6
        elevation_beam_center_probability = 1.0

        # === Sonar Physical Parameters ===
        wavelength = c / f
        k = 2 * np.pi / wavelength

        # === Elevation angles ===
        elevation_angles = np.arange(
            -vertical_beamwidth / 2,
            vertical_beamwidth / 2 + 1,
            1,
            dtype=np.float32
        )

        angles_rad = np.deg2rad(elevation_angles)

        # === Physical beam pattern (sinc) using SONAR beam directivity pattern ===
        temp = (k * aperture_size / 2) * np.sin(angles_rad)
        DI = np.ones_like(temp, dtype=np.float32)
        non_zero_mask = np.abs(temp) > 1e-10
        DI[non_zero_mask] = np.sin(temp[non_zero_mask]) / temp[non_zero_mask]

        DI_normalized = np.clip(DI, elevation_beam_edge_probability, elevation_beam_center_probability)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Beam probabilities vs elevation angle
        ax1.plot(elevation_angles, DI_normalized, 'b-o', linewidth=2, markersize=6)
        ax1.axhline(y=elevation_beam_edge_probability, color='r', linestyle='--', alpha=0.5, label=f'Min prob = {elevation_beam_edge_probability}')
        ax1.axhline(y=elevation_beam_center_probability, color='g', linestyle='--', alpha=0.5, label=f'Max prob = {elevation_beam_center_probability}')
        ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Elevation Angle (degrees)', fontsize=12)
        ax1.set_ylabel('Beam Probability', fontsize=12)
        ax1.set_title('Beam Probability vs Elevation Angle', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0.4, 1.0])

        # Plot 2: Polar representation
        angles_rad = np.deg2rad(elevation_angles)
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles_rad, DI_normalized, 'b-o', linewidth=2, markersize=6)
        ax2.set_theta_zero_location('N')
        ax2.set_title('Beam Pattern (Polar View)', fontsize=14, pad=20)
        ax2.set_ylim([0, 1.0])

        plt.tight_layout()
        plt.show()

        # Print some statistics
        print(f"Elevation angles: {elevation_angles}")
        print(f"Beam probabilities: {DI_normalized}")
        print(f"Peak probability (at 0°): {DI[len(DI)//2]:.4f}")

def plot_gaussian():
    # Your parameters
    vertical_beamwidth = 12  # example value, adjust as needed
    min_prob = 0.6
    max_prob = 1.0
    sigma = 2.0

    # Generate elevation angles
    elevation_angles = np.arange(
        -vertical_beamwidth / 2,
        vertical_beamwidth / 2 + 1,
        1,
        dtype=np.float32
    )

    # Calculate beam probabilities
    beam_probs = min_prob + (max_prob - min_prob) * np.exp(
        -0.5 * (elevation_angles / sigma) ** 2
    ).astype(np.float32)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Beam probabilities vs elevation angle
    ax1.plot(elevation_angles, beam_probs, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=min_prob, color='r', linestyle='--', alpha=0.5, label=f'Min prob = {min_prob}')
    ax1.axhline(y=max_prob, color='g', linestyle='--', alpha=0.5, label=f'Max prob = {max_prob}')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Beam Probability', fontsize=12)
    ax1.set_title('Beam Probability vs Elevation Angle', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0.4, 1.0])

    # Plot 2: Polar representation
    angles_rad = np.deg2rad(elevation_angles)
    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles_rad, beam_probs, 'b-o', linewidth=2, markersize=6)
    ax2.set_theta_zero_location('N')
    ax2.set_title('Beam Pattern (Polar View)', fontsize=14, pad=20)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Elevation angles: {elevation_angles}")
    print(f"Beam probabilities: {beam_probs}")
    print(f"Peak probability (at 0°): {beam_probs[len(beam_probs)//2]:.4f}")

if __name__ == "__main__":
      plot_directivity()
    # plot_gaussian()
