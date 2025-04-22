import matplotlib.pyplot as plt
import numpy as np
import yaml

def parse_vision_pose(file_path):
    """Parses a vision_pose.txt file and extracts time, position, and orientation data."""
    times = []
    positions = []
    orientations = []
    with open(file_path, 'r') as file:
        data_blocks = file.read().strip().split('---')
        for block in data_blocks:
            data = yaml.safe_load(block)
            if data is None:
                continue
            sec = data['header']['stamp']['sec']
            nanosec = data['header']['stamp']['nanosec']
            time = sec + nanosec * 1e-9  # Convert to seconds
            times.append(time)
            
            pos = data['pose']['position']
            ori = data['pose']['orientation']
            positions.append([pos['x'], pos['y'], pos['z']])
            orientations.append([ori['x'], ori['y'], ori['z'], ori['w']])
    return np.array(times), np.array(positions), np.array(orientations)

def plot_vision_pose(file_path):
    """Plots position and orientation of vision poses separately with time on x-axis."""
    times, positions, orientations = parse_vision_pose(file_path)
    
    # Plot positions
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(times, positions[:, 0], label='X Position', marker='o')
    axs[0].plot(times, positions[:, 1], label='Y Position', marker='o')
    axs[0].plot(times, positions[:, 2], label='Z Position', marker='o')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position')
    axs[0].set_title('Position over Time')
    axs[0].legend()
    axs[0].grid()
    
    # Plot orientations (quaternion components)
    axs[1].plot(times, orientations[:, 0], label='X Orientation', marker='o')
    axs[1].plot(times, orientations[:, 1], label='Y Orientation', marker='o')
    axs[1].plot(times, orientations[:, 2], label='Z Orientation', marker='o')
    axs[1].plot(times, orientations[:, 3], label='W Orientation', marker='o')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Orientation (Quaternion)')
    axs[1].set_title('Orientation over Time')
    axs[1].legend()
    axs[1].grid()
    
    plt.tight_layout()
    plt.show()

# Example usage:
plot_vision_pose('vicon_realsense.txt')
plot_vision_pose('realsense.txt')
