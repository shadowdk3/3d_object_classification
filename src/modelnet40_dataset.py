import os
import numpy as np
import trimesh
import torch
# import torch.nn as nn
from torch.utils.data import Dataset

# Small Dataset class
# Load ModelNet40 meshes, convert them to normalized point clouds,
# and cache the processed samples for faster training.
class ModelNet40Dataset(Dataset):
    def __init__(self, root, split="train", max_samples=1000):
        self.samples = []
        
        # get class name with the folder
        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        # create id mapping
        self.class_to_idx = {
            name: i for i, name in enumerate(self.classes)
        }

        for class_name in self.classes:
            # get "split" folder name
            folder = os.path.join(root, class_name, split)

            if not os.path.exists(folder):
                continue

            # add files name into list
            files = [
                f for f in os.listdir(folder)
                if f.endswith(".off")
            ]

            
            for f in files:
                self.samples.append(
                    (
                        os.path.join(folder, f),
                        self.class_to_idx[class_name]
                    )
                )

        if max_samples > -1:
            # Keep it small
            self.samples = self.samples[:max_samples]
        
        # Cache
        self.cache = {}
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        if index in self.cache:
            return self.cache[index]
            
        # get the samples and seperate into file path and class id 
        path, label = self.samples[index]

        # load .off file as a mesh
        mesh = trimesh.load(path, force="mesh")

        # Check the mesh
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(
                f"Invalid vertices found in: {path}"
            )
    
        # Sample 1024 points
        points, _ = trimesh.sample.sample_surface(
            mesh, 1024
        )

        # Check for NaN or Inf
        if not np.isfinite(points).all():
            print("Invalid points:", path)
    
        # Normalize
        points = points - points.mean(axis=0)

        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale

        # convert points to tensor
        points = torch.tensor(
            points,
            dtype=torch.float32
        )

        # conert label to tensor
        label = torch.tensor(
            label,
            dtype=torch.long
        )

        self.cache[index] = (points, label)

        return points, label