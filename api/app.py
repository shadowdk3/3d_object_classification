import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import predictor
import numpy as np

# Page configuration
st.set_page_config(
    page_title="PointNet 3D Classifier",
    page_icon="🧊",
    layout="wide",
)

st.title("PointNet 3D Object Classifier")
st.write(
    "Upload a 3D mesh to preview and classify its object category."
)

# File upload
uploaded_file = st.file_uploader(
    "Choose a 3D model (.off)",
    type=["off"],
)

if uploaded_file is not None:
    
    content = uploaded_file.getvalue()
    
    # Prediction (above graph)
    if st.button("Predict Object Class", type="primary"):
        with st.spinner("Running PointNet inference..."):
            result = predictor.predict(content)

        st.session_state["prediction_result"] = result

    # Display prediction result
    if "prediction_result" in st.session_state:
        st.success(
            f"Predicted Class: "
            f"{st.session_state['prediction_result']['predicted_class']}"
        )

    # Display interactive 3D preview
    st.subheader("3D Mesh Preview")
    
    # Load the mesh
    mesh = predictor.load_mesh(content)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if len(vertices) > 0 and len(faces) > 0:
        # Ensure all faces contain valid vertex indices
        valid_faces = (
            (faces >= 0).all()
            and (faces < len(vertices)).all()
        )

        if valid_faces:
            fig = go.Figure(
                data=[
                    go.Mesh3d(
                        x=vertices[:, 0].tolist(),
                        y=vertices[:, 1].tolist(),
                        z=vertices[:, 2].tolist(),
                        i=faces[:, 0].tolist(),
                        j=faces[:, 1].tolist(),
                        k=faces[:, 2].tolist(),
                        color="lightblue",
                        opacity=1.0,
                    )
                ]
            )

            fig.update_layout(
                height=600,
                scene=dict(aspectmode="data"),
                margin=dict(l=0, r=0, t=20, b=0),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="uploaded_mesh",
            )
        else:
            st.error("The mesh contains invalid face indices.")
            st.stop()
    else:
        st.error("The mesh has no vertices or faces.")
        st.stop()
    
    st.divider()
