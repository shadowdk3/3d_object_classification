import onnxruntime as ort
import utils.paths
import os
import io
import numpy as np
import trimesh

class PointNetPredictor:
    def __init__(self):
        self.class_names = sorted([
            d for d in os.listdir(utils.paths.DATA_PATH)
            if os.path.isdir(os.path.join(utils.paths.DATA_PATH, d))
        ])
        
        # Load the ONNX model once
        self.session = ort.InferenceSession(
            utils.paths.BEST_ONNX_PATH,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("ONNX model loaded successfully!")
        print("Number of classes:", len(self.class_names))
        print("Providers:", self.session.get_providers())

    def preprocess(self, content: bytes) -> np.ndarray: 
        """Convert OFF file bytes into a normalized point cloud.""" 
        mesh = trimesh.load( 
            io.BytesIO(content), 
            file_type="off", 
            force="mesh", ) 

        # Sample 1024 points from mesh surface 
        points, _ = trimesh.sample.sample_surface(mesh, 1024) 
        
        points = np.asarray(points, dtype=np.float32) 
        
        # Normalize using the training preprocessing 
        points = points - points.mean(axis=0) 
        scale = np.max(np.linalg.norm(points, axis=1)) 
        
        if scale > 0: 
            points = points / scale 
            
        # Shape: [1, 1024, 3] 
        return np.expand_dims(points, axis=0).astype(np.float32)
    
    def predict(self, content: bytes) -> dict: 
        """Predict the class of an uploaded OFF file.""" 
        points_input = self.preprocess(content) 
        
        output = self.session.run( 
            [self.output_name], 
            {self.input_name: points_input}, 
        )[0] 
        
        prediction = int(np.argmax(output, axis=1)[0]) 
        return { 
            "predicted_index": prediction, 
            "predicted_class": self.class_names[prediction], 
        }
    
    def load_mesh(self, content: bytes):
        """Load an OFF file into a Trimesh object."""
        return trimesh.load(
            io.BytesIO(content),
            file_type="off",
            force="mesh",
            process=False,
        )
    
    def show_mesh(self, mesh):
        """Open the mesh in a separate 3D viewer."""
        mesh.show()
        
# Create one predictor instance when the module is imported 
predictor = PointNetPredictor()