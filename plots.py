import numpy as np
import matplotlib.pyplot as plt

def events_to_voxel_grid(events, num_bins, width, height):
    width = int(width)
    height = int(height)
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float64).ravel()

    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp
    deltaT = deltaT if deltaT != 0 else 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    return voxel_grid

def visualize_voxel_grid(voxel_grid):
    num_bins, height, width = voxel_grid.shape

    fig, axes = plt.subplots(1, num_bins, figsize=(15, 5))
    fig.suptitle('Voxel Grid Visualization')

    for i in range(num_bins):
        ax = axes[i]
        heatmap = ax.imshow(voxel_grid[i], cmap='viridis', aspect='auto')
        ax.set_title(f'Bin {i+1}')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        fig.colorbar(heatmap, ax=ax)

    plt.tight_layout()
    plt.show()

# Example usage with random events data
def generate_random_events(num_events, width, height):
    np.random.seed(42)
    timestamps = np.sort(np.random.rand(num_events))
    x_coords = np.random.randint(0, width, num_events)
    y_coords = np.random.randint(0, height, num_events)
    polarities = np.random.randint(0, 2, num_events)
    events = np.column_stack((timestamps, x_coords, y_coords, polarities))
    return events

# Parameters
num_bins = 5
width = 10
height = 10
num_events = 1000

# Generate random events and create voxel grid
events = generate_random_events(num_events, width, height)
voxel_grid = events_to_voxel_grid(events, num_bins, width, height)

# Visualize the voxel grid
visualize_voxel_grid(voxel_grid)
