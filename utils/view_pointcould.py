import os
import trimesh
import utils.paths
import matplotlib.pyplot as plt

file_path = os.path.join(utils.paths.DATA_PATH, "airplane", "train", "airplane_0007.off")
print(file_path)

# display data with trimesh
mesh = trimesh.load(file_path)

#Convert it to 1024 points
points, _ = trimesh.sample.sample_surface(mesh, 1024)
print(points.shape)

# Visualize the point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    s=3
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()