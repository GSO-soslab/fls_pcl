# fls_ism

Inverse Sensor Model for Forward-Looking Sonar (FLS). Converts raw sonar polar images into probabilistic point clouds and voxel-based occupancy maps for AUV perception.

## Dependencies

```bash
sudo apt-get install ros-jazzy-image-transport ros-jazzy-cv-bridge ros-jazzy-tf2-ros libboost-all-dev ros-jazzy-tf-transformations
pip install scipy
```

## Nodes

### `fls_pcl_node` — `fls_ism/fls_pcl.py`

Main processing node. Subscribes to raw sonar images and outputs probabilistic point clouds.

**Subscribes**
| Type | Description |
|---|---|
| `Image` | Raw polar sonar image |
| `Ping` | Sonar metadata (bearings, range) — real hardware only |
| `Marker` | Voxel FOV geometry from `fls_voxel_node` |

**Publishes**
| Type | Description |
|---|---|
| `PointCloud2` | Voxel-based probabilistic cloud (intensity = occupancy probability) |
| `PointCloud2` | Peak intensity per azimuth beam |
| `PointCloud2` | Unfiltered sonar image as points |
| `Image` | Preprocessed sonar image |

**Pipeline**
1. Preprocesses image: tapered wedge mask + anisotropic diffusion (Perona-Malik) for real hardware; passthrough in sim mode.
2. Computes bearing directivity weights using a sinc beam pattern over the vertical FOV.
3. Maps pixel intensity to occupancy probability (linear interpolation between configured bounds).
4. Max-pools measurements onto the pre-computed voxel grid and applies elevation weights.
5. Transforms to world frame via TF2 and filters by depth.

---

### `fls_voxel_node` — `fls_ism/fls_voxels.py`

Publishes the sonar FOV geometry as a voxel marker grid. Runs once on startup with a latching QoS so downstream nodes receive it on connect.

**Publishes**
| Type | Description |
|---|---|
| `Marker` | 3D voxel grid covering the sonar cone |

---


## Launch Files

### `fls_ism.launch.py`
Starts `fls_voxel_node` and `fls_pcl_node` under the `alpha_rise` namespace. Primary launch file for online sonar processing (live hardware or bag replay).

```bash
ros2 launch fls_ism fls_ism.launch.py
```

### `post_process.launch.py`
Full post-processing stack for bag replay. Launches:
- FLS processing (`fls_ism.launch.py`)
- MSIS voxel processing (`pcl_proc`)
- Voxel log-odds accumulation (`pcl_proc`)
- MBES inverse sensor model (`mbes_ism`)
- Vehicle description + path (`alpha_rise_bringup`)
- RViz (`config_post.rviz`)
- `ros2 bag play` with `use_sim_time: true`

---

## Configuration

### `config/fls_params.yaml`
Shared parameters for `fls_pcl_node` and `fls_voxel_node`.

| Parameter | Default | Description |
|---|---|---|
| `sim` | `false` | Simulation mode — skips image preprocessing, derives bearings from image width |
| `vertical_fov_deg` | `12.0` | Vertical beam FOV (degrees) |
| `horizontal_fov_deg` | `70.0` | Horizontal FOV (degrees) |
| `range_max` | `40.0` | Max sensor range (m) |
| `resolution` | `1.0` | Voxel side length (m) |
| `threshold_intensity` | `30` | Minimum pixel intensity (0–255) |
| `threshold_min_range` | `5.0` | Minimum range filter (m) |
| `max_depth` / `min_depth` | `-1.0` / `-15.0` | World-frame depth bounds (ENU, m) |
| `lower_bound_intensity` | `56` | Intensity mapped to `min_probability` |
| `upper_bound_intensity` | `100` | Intensity mapped to `max_probability` |
| `min_probability` | `0.2` | Minimum occupancy probability output |
| `max_probability` | `0.9` | Maximum occupancy probability output |
| `frequency` | `1.2e6` | Sonar frequency (Hz) |
| `sound_speed` | `1500` | Speed of sound (m/s) |
| `aperture_size` | `0.0067` | Aperture size for sinc beam directivity (m) |
| `save_as_pcd` | `false` | Accumulate and save cloud to PCD on shutdown |

---
