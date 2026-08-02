import runpod
import os
import tempfile
import base64
from PIL import Image
import torch
from trellis.pipelines import TrellisImageTo3DPipeline

# Initialize model at container start to avoid cold-start overhead per request
print("Loading TRELLIS model into VRAM...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
print("Model loaded successfully.")

def handler(job):
    job_input = job.get('input', {})
    prompt = job_input.get('prompt')
    resolution = job_input.get('octree_resolution', 'low')
    
    if not prompt:
        return {"error": "Missing 'prompt' in input."}
    
    try:
        image_url = job_input.get('image_url')
        if not image_url:
            return {"error": "Missing 'image_url' to generate 3D mesh from."}
            
        import requests
        from io import BytesIO
        
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Run inference
        print(f"Running inference at {resolution} resolution...")
        outputs = pipeline.run(
            img,
            seed=42,
            sparse_structure_sampler_params={"steps": 12 if resolution == 'low' else 25},
            slat_sampler_params={"steps": 12 if resolution == 'low' else 25}
        )
        
        # Save GLB to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            glb_path = tmp.name
            
        # TRELLIS exports a Gaussian/Mesh structure, extract the mesh and save
        from trellis.utils import render_utils, postprocessing_utils
        mesh = postprocessing_utils.to_mesh(outputs)
        mesh.export(glb_path)
        
        # Read file as base64
        with open(glb_path, "rb") as f:
            glb_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        os.remove(glb_path)
        
        # Return the base64 string directly to the Cloudflare Worker
        return {
            "mesh_base64": glb_base64,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
