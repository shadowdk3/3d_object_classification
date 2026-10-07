import os
import trimesh
import utils.paths

file_path = os.path.join(utils.paths.DATA_PATH, "airplane", "train", "airplane_0007.off")
print(file_path)

# display data with trimesh
mesh = trimesh.load(file_path)
mesh.show()