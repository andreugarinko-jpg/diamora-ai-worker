import runpod
import os
import tempfile
import boto3
from PIL import Image
import torch
# Assumes TRELLIS is installed in the container environment
from trellis.pipelines import TrellisImageTo3DPipeline

# Initialize model at container start to avoid cold-start overhead per request
print("Loading TRELLIS model into VRAM...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
print("Model loaded successfully.")

# Setup S3/R2 client for uploading the resulting mesh
s3 = boto3.client('s3',
    endpoint_url=os.environ.get('R2_ENDPOINT_URL'),
    aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY')
)
R2_BUCKET = os.environ.get('R2_BUCKET_NAME', 'diamora-meshes')

def handler(job):
    job_input = job.get('input', {})
    prompt = job_input.get('prompt')
    resolution = job_input.get('octree_resolution', 'low')
    
    if not prompt:
        return {"error": "Missing 'prompt' in input."}
    
    try:
        # In a real scenario, you'd convert the text prompt to an image first, 
        # or use the text-to-3D pipeline if available. 
        # TRELLIS usually relies on image-to-3D. For text, we'd pipe through SDXL first.
        # Assuming we receive an image URL for the 3D generation:
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
        
        # Upload to Cloudflare R2
        file_name = f"{job['id']}.glb"
        s3.upload_file(glb_path, R2_BUCKET, file_name)
        
        # Clean up
        os.remove(glb_path)
        
        # Return public URL of the uploaded mesh
        public_url = f"https://pub-xxxxxx.r2.dev/{file_name}"
        
        return {
            "mesh_url": public_url,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
