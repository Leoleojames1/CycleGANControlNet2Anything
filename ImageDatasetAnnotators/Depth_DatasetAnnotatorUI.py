import gradio as gr
import cv2
import numpy as np
import torch
import os
import sys
import shutil
import tempfile
from PIL import Image
from huggingface_hub import HfApi, HfFolder, hf_hub_download, create_repo
import time
import random
import threading
from datetime import datetime, timedelta
from tqdm import tqdm
import concurrent.futures
import traceback
import os
import sys

# Print all environment variables to check
print("All environment variables:")
for key, value in os.environ.items():
    if "DEPTH" in key:
        print(f"{key}: {value}")

# Check specific variable
depth_path = os.getenv('DEPTH_ANYTHING_V2_PATH')
print(f"DEPTH_ANYTHING_V2_PATH value: {depth_path}")

# Continue with your code
if depth_path is None:
    depth_anything_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Environment variable not set. Using current directory: {depth_anything_path}")
else:
    depth_anything_path = depth_path
    print(f"Using environment variable path: {depth_anything_path}")

sys.path.append(depth_anything_path)
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print("Successfully imported DepthAnythingV2")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Contents of directory: {os.listdir(depth_anything_path)}")
    if os.path.exists(os.path.join(depth_anything_path, 'depth_anything_v2')):
        print(f"Contents of depth_anything_v2: {os.listdir(os.path.join(depth_anything_path, 'depth_anything_v2'))}")

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder2name = {
    'vits': 'Small',
    'vitb': 'Base',
    'vitl': 'Large'
}

name2encoder = {v: k for k, v in encoder2name.items()}

# Model IDs and filenames for HuggingFace Hub
MODEL_INFO = {
    'vits': {
        'repo_id': 'depth-anything/Depth-Anything-V2-Small',
        'filename': 'depth_anything_v2_vits.pth'
    },
    'vitb': {
        'repo_id': 'depth-anything/Depth-Anything-V2-Base',
        'filename': 'depth_anything_v2_vitb.pth'
    },
    'vitl': {
        'repo_id': 'depth-anything/Depth-Anything-V2-Large',
        'filename': 'depth_anything_v2_vitl.pth'
    }
}

# Global variables for model management
current_model = None
current_encoder = None

# Global variable for live preview
live_preview_queue = []
live_preview_lock = threading.Lock()

def download_model(encoder):
    """Download the specified model from HuggingFace Hub"""
    model_info = MODEL_INFO[encoder]
    
    # Check if the file already exists in the checkpoints directory of DEPTH_ANYTHING_V2_PATH
    depth_path = os.getenv('DEPTH_ANYTHING_V2_PATH')
    if depth_path:
        checkpoint_dir = os.path.join(depth_path, 'checkpoints')
        local_file = os.path.join(checkpoint_dir, model_info['filename'])
        if os.path.exists(local_file):
            print(f"Using existing model file: {local_file}")
            return local_file
    
    # If not found, download it
    model_path = hf_hub_download(
        repo_id=model_info['repo_id'],
        filename=model_info['filename'],
        local_dir='checkpoints'
    )
    return model_path

def load_model(encoder):
    """Load the specified model"""
    global current_model, current_encoder
    if current_encoder != encoder:
        model_path = download_model(encoder)
        current_model = DepthAnythingV2(**model_configs[encoder])
        current_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        current_model = current_model.to(DEVICE).eval()
        current_encoder = encoder
    return current_model

def convert_to_bw(image):
    """Convert image to black and white"""
    if isinstance(image, Image.Image):
        return image.convert('L').convert('RGB')
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return image

def blend_images(original, depth_colored, opacity=0.5, make_bw=False, depth_on_top=True):
    """Blend original image with depth map using specified opacity
    opacity: 0.0 = original image only, 1.0 = depth map only
    depth_on_top: If True, depth map is blended on top of original image"""
    
    # Convert inputs to numpy arrays if needed
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(depth_colored, Image.Image):
        depth_colored = np.array(depth_colored)
    
    # Convert original to black and white if requested
    if make_bw:
        original = cv2.cvtColor(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    
    # Ensure both images are float32 for blending
    original = original.astype(np.float32)
    depth_colored = depth_colored.astype(np.float32)
    
    # Calculate blend based on opacity
    if depth_on_top:
        blended = original * (1 - opacity) + depth_colored * opacity
    else:
        blended = original * opacity + depth_colored * (1 - opacity)
    
    # Clip values and convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended  # Return numpy array instead of PIL Image

@torch.inference_mode()
def predict_depth(image, encoder, invert_depth=False):
    """Predict depth using the selected model"""
    model = load_model(encoder)
    if model is None:
        raise ValueError(f"Model for encoder {encoder} could not be loaded.")
    
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Get depth prediction
    depth = model.infer_image(image)
    
    # Ensure we have valid depth values (no NaNs or infs)
    depth = np.nan_to_num(depth)
    
    # Normalize to 0-255 range for visualization
    depth_min = depth.min()
    depth_max = depth.max()
    
    if depth_max > depth_min:
        # Linear normalization
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        # Apply slight gamma correction to enhance visibility
        depth_normalized = np.power(depth_normalized, 0.8)
        # Scale to 0-255 range
        depth_map = (depth_normalized * 255).astype(np.uint8)
    else:
        depth_map = np.zeros_like(depth, dtype=np.uint8)
    
    # Invert if requested (after normalization)
    if invert_depth:
        depth_map = 255 - depth_map
    
    return depth_map

def apply_colormap(depth, colormap=cv2.COLORMAP_TURBO, reverse_colormap=False):
    """Apply a colormap to the depth image"""
    # Ensure input is a valid numpy array
    if not isinstance(depth, np.ndarray):
        depth = np.array(depth)
    
    # Ensure single channel
    if len(depth.shape) > 2:
        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
    
    # Reverse depth values if requested
    if reverse_colormap:
        depth = 255 - depth
    
    # Apply colormap
    colored = cv2.applyColorMap(depth, colormap)
    
    # Convert BGR to RGB
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored_rgb

def resize_image(image, max_size=1200):
    """Resize image if its dimensions exceed max_size"""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image

def save_image(image, path):
    """Save PIL Image to the specified path"""
    image.save(path, format="PNG")

def add_to_live_preview(original_image, depth_image):
    """Add processed images to the live preview queue"""
    global live_preview_queue
    with live_preview_lock:
        # Keep only the most recent 10 pairs
        if len(live_preview_queue) >= 10:
            live_preview_queue.pop(0)
        live_preview_queue.append([original_image, depth_image])

def get_live_preview():
    """Get the current live preview images"""
    global live_preview_queue
    with live_preview_lock:
        return live_preview_queue.copy()

class ProcessProgressTracker:
    """Track progress of image processing"""
    def __init__(self, total_files):
        self.total_files = total_files
        self.processed_files = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update(self):
        with self.lock:
            self.processed_files += 1
            elapsed = time.time() - self.start_time
            files_per_sec = self.processed_files / elapsed if elapsed > 0 else 0
            eta = (self.total_files - self.processed_files) / files_per_sec if files_per_sec > 0 else 0
            
            # Only print status every 5 files or at completion
            if self.processed_files % 5 == 0 or self.processed_files == self.total_files:
                print(f"Processed {self.processed_files}/{self.total_files} images " +
                      f"({self.processed_files/self.total_files*100:.1f}%) " +
                      f"- {files_per_sec:.2f} imgs/sec - ETA: {timedelta(seconds=int(eta))}")
            
            return self.processed_files, self.total_files

def process_image(args):
    """Process a single image for multi-threading"""
    filename, folder_path, temp_dir, output_dir, encoder, progress_tracker, invert_depth, colormap, enable_blending, blend_opacity, make_base_bw, depth_on_top, use_colormap, reverse_colormap = args
    
    try:
        image_path = os.path.join(folder_path, filename)
        
        # Define output paths
        temp_image_path = os.path.join(temp_dir, filename)
        output_image_path = os.path.join(output_dir, filename) if output_dir else None
        
        # Process image
        image = Image.open(image_path).convert('RGB')
        image = resize_image(image)
        image_np = np.array(image)
        
        # Generate depth map
        depth_map = predict_depth(image_np, encoder, invert_depth)
        
        # Handle colormap and depth visualization
        if use_colormap:
            final_output = apply_colormap(depth_map, colormap, reverse_colormap)
        else:
            final_output = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        
        # Handle blending if enabled
        if enable_blending:
            final_output = blend_images(
                image_np, 
                final_output, 
                opacity=blend_opacity,
                make_bw=make_base_bw,
                depth_on_top=depth_on_top
            )
        
        final_output = Image.fromarray(final_output)
        
        # Create depth filename
        base, ext = os.path.splitext(filename)
        depth_filename = f"{base}_depth{ext}"
        
        # Save to temp dir
        temp_depth_path = os.path.join(temp_dir, depth_filename)
        save_image(Image.fromarray(image_np), temp_image_path)
        save_image(final_output, temp_depth_path)
        
        # Save to output dir if specified
        if output_dir:
            output_depth_path = os.path.join(output_dir, depth_filename)
            save_image(Image.fromarray(image_np), output_image_path)
            save_image(final_output, output_depth_path)
        
        # Update live preview
        add_to_live_preview(Image.fromarray(image_np), final_output)
        
        # Update progress
        progress_tracker.update()
        
        return temp_image_path, temp_depth_path
    except Exception as e:
        print(f"ERROR processing image {filename}: {e}")
        traceback.print_exc()
        return None, None

def process_images(folder_path, encoder, output_dir=None, max_workers=1, invert_depth=False, 
                  colormap=cv2.COLORMAP_TURBO, enable_blending=False, blend_opacity=0.0, 
                  make_base_bw=False, depth_on_top=True, use_colormap=True, reverse_colormap=False):
    """Process all images in the folder and generate depth maps"""
    images = []
    depth_maps = []
    temp_dir = tempfile.mkdtemp()
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Clear previous live preview
    global live_preview_queue
    with live_preview_lock:
        live_preview_queue = []
    
    # Validate folder path
    print(f"Checking folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder path does not exist: {folder_path}")
        return images, depth_maps, temp_dir
    
    if not os.path.isdir(folder_path):
        print(f"ERROR: Path is not a directory: {folder_path}")
        return images, depth_maps, temp_dir
    
    # List files and check for images
    try:
        all_files = os.listdir(folder_path)
        print(f"Found {len(all_files)} items in folder")
        
        # Count image files, excluding depth maps
        image_files = [f for f in all_files 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                      and not f.lower().endswith('_depth.png')
                      and not f.lower().endswith('_depth.jpg')
                      and not f.lower().endswith('_depth.jpeg')]
        
        print(f"Found {len(image_files)} original image files (excluding depth maps)")
        
        if len(image_files) == 0:
            print("WARNING: No valid image files found in the specified folder")
            print("Allowed extensions are: .png, .jpg, .jpeg")
            # Print first 10 files to help debugging
            if all_files:
                print("First 10 files in directory:")
                for f in all_files[:10]:
                    print(f"  - {f}")
            return images, depth_maps, temp_dir
            
    except Exception as e:
        print(f"ERROR accessing folder: {e}")
        return images, depth_maps, temp_dir
    
    # Setup progress tracking
    progress_tracker = ProcessProgressTracker(len(image_files))
    
    # Process images in parallel if using GPU
    if DEVICE == 'cuda' and max_workers > 1:
        print(f"Processing images with {max_workers} workers...")
        
        # Fix process_args creation
        process_args = [(
            filename, folder_path, temp_dir, output_dir, encoder, 
            progress_tracker, invert_depth, colormap, enable_blending,
            blend_opacity, make_base_bw, depth_on_top, use_colormap, reverse_colormap
        ) for filename in image_files]
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_image, process_args))
            
            # Filter out any None results from errors
            valid_results = [(img, depth) for img, depth in results if img is not None]
            
            if valid_results:
                images, depth_maps = zip(*valid_results)
                images = list(images)
                depth_maps = list(depth_maps)
    else:
        # Process sequentially
        print("Processing images sequentially...")
        for filename in image_files:
            result = process_image((filename, folder_path, temp_dir, output_dir, encoder, progress_tracker, invert_depth, colormap, enable_blending, blend_opacity, make_base_bw, depth_on_top, use_colormap))
            if result[0] is not None:
                images.append(result[0])
                depth_maps.append(result[1])
    
    print(f"Successfully processed {len(images)} images")
    return images, depth_maps, temp_dir

def exponential_backoff(retry_count, base_wait=30):
    """Calculate wait time with exponential backoff and jitter"""
    wait_time = min(base_wait * (2 ** retry_count), 3600)  # Cap at 1 hour
    jitter = random.uniform(0.8, 1.2)  # Add 20% jitter
    return wait_time * jitter

def safe_upload_file(api, path_or_fileobj, path_in_repo, repo_id, token, max_retries=5):
    """Upload a file with retry logic for rate limiting"""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                repo_type="dataset"
            )
            return True
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "rate-limited" in error_str:
                # Progressive backoff strategy - wait longer with each retry
                wait_time = (5 + retry_count * 5) * 60  # 5, 10, 15, 20, 25 minutes
                
                retry_count += 1
                print(f"Rate limited! Waiting for {wait_time/60:.1f} minutes before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            else:
                # For non-rate limit errors, raise the exception
                print(f"Error uploading file: {e}")
                raise e
    
    print(f"Failed to upload after {max_retries} retries: {path_in_repo}")
    return False

def create_resume_file(resume_dir, all_files, start_idx, repo_id):
    """Create a resume file to continue uploads later"""
    os.makedirs(resume_dir, exist_ok=True)
    resume_path = os.path.join(resume_dir, f"resume_{repo_id.replace('/', '_')}.txt")
    
    with open(resume_path, "w") as f:
        # Format: current_index, total_files, datetime
        f.write(f"{start_idx},{len(all_files)},{datetime.now().isoformat()}\n")
        
        # Write remaining files to upload
        for idx in range(start_idx, len(all_files)):
            file_path, file_name, file_type = all_files[idx]
            f.write(f"{file_path}|{file_name}|{file_type}\n")
    
    return resume_path

def upload_to_hf(images, depth_maps, repo_id, break_every=100, resume_dir="upload_resume", resume_file=None):
    """Upload images and depth maps to Hugging Face Hub with regular breaks"""
    api = HfApi()
    token = HfFolder.get_token()
    
    # Create combined list of files to upload
    all_files = []
    
    # If resuming from file, read the list of files to upload
    start_idx = 0
    
    if resume_file and os.path.exists(resume_file):
        print(f"Resuming upload from {resume_file}")
        with open(resume_file, "r") as f:
            lines = f.readlines()
            header = lines[0].strip().split(",")
            start_idx = int(header[0])
            
            # Read file entries
            for line in lines[1:]:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    all_files.append((parts[0], parts[1], parts[2]))
        
        print(f"Resuming upload from index {start_idx}, {len(all_files)} files remaining")
    else:
        # Create new file list
        for i, (image_path, depth_map_path) in enumerate(zip(images, depth_maps)):
            all_files.append((image_path, os.path.basename(image_path), f"pair_{i+1}_image"))
            all_files.append((depth_map_path, os.path.basename(depth_map_path), f"pair_{i+1}_depth"))
    
    total_files = len(all_files)
    
    # Validate break interval
    if break_every <= 0:
        break_every = 100
    
    # Create resume file
    resume_path = create_resume_file(resume_dir, all_files, start_idx, repo_id)
    print(f"Created resume file: {resume_path}")
    print(f"If the upload is interrupted, you can resume using this path in the UI")
    
    # Ensure the repository exists and is of type 'dataset'
    try:
        api.repo_info(repo_id=repo_id, token=token)
    except Exception as e:
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", token=token)
        except Exception as create_e:
            if "You already created this dataset repo" not in str(create_e):
                raise create_e
    
    print(f"Beginning upload of {total_files} files (starting at {start_idx+1})")
    print(f"Will take a 3-minute break after every {break_every} files to avoid rate limiting")
    
    # Track upload metrics
    upload_start_time = time.time()
    success_count = 0
    
    # Create progress bar
    progress_bar = tqdm(total=total_files, initial=start_idx, desc="Uploading", 
                       unit="files", dynamic_ncols=True)
    
    try:
        # Process files with periodic breaks
        for idx in range(start_idx, total_files):
            file_path, file_name, file_type = all_files[idx]
            
            # Take a break every break_every files (but not at the start)
            if idx > start_idx and (idx - start_idx) % break_every == 0:
                break_minutes = 3
                
                # Longer break after known problematic thresholds
                if idx >= 125 and idx < 130:
                    break_minutes = 15
                    tqdm.write(f"===== EXTENDED RATE LIMIT PREVENTION BREAK =====")
                    tqdm.write(f"Approaching critical threshold (files 125-130). Taking a longer {break_minutes}-minute break...")
                else:
                    tqdm.write(f"===== RATE LIMIT PREVENTION BREAK =====")
                    tqdm.write(f"Uploaded {break_every} files. Taking a {break_minutes}-minute break...")
                
                create_resume_file(resume_dir, all_files, idx, repo_id)
                
                # Show countdown timer for the break
                for remaining in range(break_minutes * 60, 0, -10):
                    mins = remaining // 60
                    secs = remaining % 60
                    tqdm.write(f"Resuming in {mins}m {secs}s...")
                    time.sleep(10)
                
                tqdm.write("Break finished, continuing uploads...")
            
            # Upload the file
            tqdm.write(f"Uploading file {idx+1}/{total_files}: {file_name}")
            success = safe_upload_file(api, file_path, file_name, repo_id, token)
            
            if not success:
                tqdm.write(f"Failed to upload {file_name} after multiple retries.")
                # Update resume file with current position
                create_resume_file(resume_dir, all_files, idx, repo_id)
                return False
            
            # Update tracking
            success_count += 1
            progress_bar.update(1)
            
            # Update resume file every 10 uploads
            if (idx + 1) % 10 == 0:
                create_resume_file(resume_dir, all_files, idx + 1, repo_id)
    
    except KeyboardInterrupt:
        print("\nUpload interrupted! Creating resume file to continue later...")
        create_resume_file(resume_dir, all_files, idx, repo_id)
        return False
    
    finally:
        progress_bar.close()
    
    # Calculate stats
    total_time = time.time() - upload_start_time
    files_per_second = success_count / total_time if total_time > 0 else 0
    
    print(f"\nUpload completed! {success_count} files uploaded in {timedelta(seconds=int(total_time))}")
    print(f"Average upload rate: {files_per_second:.2f} files/sec")
    
    return True

def process_and_upload(folder_path, model_name, invert_depth, colormap_name, output_dir, 
                      upload_to_hf_toggle, repo_id, break_every=100, parallel_workers=1, 
                      resume_file=None, enable_blending=False, blend_opacity=0.0, 
                      make_base_bw=False, depth_on_top=True, use_colormap=True, reverse_colormap=False):
    """Process images and upload them to Hugging Face or save locally"""
    encoder = name2encoder[model_name]
    colormap = get_colormap_by_name(colormap_name)
    
    # If resume file is provided, only upload (skip processing)
    if resume_file and os.path.exists(resume_file) and upload_to_hf_toggle:
        print(f"Resuming upload from file: {resume_file}")
        success = upload_to_hf([], [], repo_id, break_every=break_every, resume_file=resume_file)
        return "Resume upload completed successfully" if success else "Resume upload was interrupted or failed"
    
    # Process images
    images, depth_maps, temp_dir = process_images(
        folder_path, 
        encoder, 
        output_dir=output_dir, 
        max_workers=parallel_workers,
        invert_depth=invert_depth,
        colormap=colormap,
        enable_blending=enable_blending,
        blend_opacity=blend_opacity,
        make_base_bw=make_base_bw,
        depth_on_top=depth_on_top,
        use_colormap=use_colormap,
        reverse_colormap=reverse_colormap
    )
    
    if not images:
        return "No images were processed. Check the logs for details."
    
    # Upload to HF if selected
    if upload_to_hf_toggle and repo_id:
        success = upload_to_hf(images, depth_maps, repo_id, break_every=break_every)
        upload_status = f"Upload {'completed successfully' if success else 'was interrupted or failed'}. "
    else:
        upload_status = ""
    
    # Output status
    if output_dir:
        local_status = f"Images and depth maps saved to {output_dir}. "
    else:
        local_status = ""
    
    # Clean up
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temp directory: {e}")
    
    return f"{local_status}{upload_status}Successfully processed {len(images)} images."

def colormap_list():
    """Get list of available OpenCV colormaps"""
    return [
        "TURBO", "JET", "PARULA", "HOT", "WINTER", "RAINBOW", 
        "OCEAN", "SUMMER", "SPRING", "COOL", "HSV", 
        "PINK", "BONE", "VIRIDIS", "PLASMA", "INFERNO"
    ]

def get_colormap_by_name(name):
    """Convert colormap name to OpenCV enum"""
    colormap_mapping = {
        "TURBO": cv2.COLORMAP_TURBO,
        "JET": cv2.COLORMAP_JET,
        "PARULA": cv2.COLORMAP_PARULA,
        "HOT": cv2.COLORMAP_HOT,
        "WINTER": cv2.COLORMAP_WINTER,
        "RAINBOW": cv2.COLORMAP_RAINBOW,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "SUMMER": cv2.COLORMAP_SUMMER,
        "SPRING": cv2.COLORMAP_SPRING,
        "COOL": cv2.COLORMAP_COOL,
        "HSV": cv2.COLORMAP_HSV,
        "PINK": cv2.COLORMAP_PINK,
        "BONE": cv2.COLORMAP_BONE,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "INFERNO": cv2.COLORMAP_INFERNO
    }
    return colormap_mapping.get(name, cv2.COLORMAP_TURBO)

def visualize_process(folder_path, model_name, invert_depth, colormap_name, 
                     blend_opacity=0.0, make_base_bw=False, depth_on_top=True, sample_count=10):
    """Process a sample of images from the folder and visualize them"""
    encoder = name2encoder[model_name]
    colormap = get_colormap_by_name(colormap_name)
    
    # Validate folder path
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return []
    
    # Get image files
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        return []
    
    # Take a sample of images
    if len(image_files) > sample_count:
        image_files = random.sample(image_files, sample_count)
    
    # Process images
    temp_dir = tempfile.mkdtemp()
    visualization = []
    
    for filename in image_files:
        try:
            image_path = os.path.join(folder_path, filename)
            temp_image_path = os.path.join(temp_dir, filename)
            shutil.copy(image_path, temp_image_path)
            
            image = Image.open(temp_image_path).convert('RGB')
            image = resize_image(image)
            image_np = np.array(image)
            
            depth_map = predict_depth(image_np, encoder, invert_depth, blend_opacity, make_base_bw)
            depth_map_colored = apply_colormap(depth_map, colormap)
            
            depth_map_path = os.path.join(temp_dir, f"depth_{filename}")
            save_image(Image.fromarray(depth_map_colored), depth_map_path)
            
            visualization.append([image, Image.fromarray(depth_map_colored)])
            print(f"Previewed {filename}")
        except Exception as e:
            print(f"Error processing image for preview: {e}")
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
        
    return visualization

def update_live_preview():
    """Update the live preview gallery"""
    return get_live_preview()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🩻 Enhanced Depth Map Generation 🩻")
    
    with gr.Tab("Generate Depth Maps"):
        folder_input = gr.Textbox(label="Folder Path", placeholder="Enter the path to the folder with images")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["Small", "Base", "Large"],
                value="Small",
                label="Model Size (Small=Fastest, Large=Best Quality)"
            )
            
            parallel_workers = gr.Slider(
                minimum=1,
                maximum=8,
                value=1 if DEVICE == 'cpu' else 2,
                step=1,
                label="Parallel Workers (GPU only)"
            )
        
        with gr.Row():
            invert_depth = gr.Checkbox(label="Invert Depth Map", value=False)
            use_colormap = gr.Checkbox(label="Use Colormap", value=True)
            reverse_colormap = gr.Checkbox(label="Reverse Colormap", value=False)
            colormap_dropdown = gr.Dropdown(
                choices=colormap_list(),
                value="TURBO",
                label="Colormap Style",
                interactive=True
            )
        
        use_colormap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_colormap],
            outputs=colormap_dropdown
        )

        with gr.Accordion("Blending Options", open=False):
            with gr.Row():
                enable_blending = gr.Checkbox(
                    label="Enable Blending",
                    value=False,
                    info="Blend depth map with original image"
                )
                make_base_bw = gr.Checkbox(
                    label="Make Original B&W",
                    value=False,
                    visible=False
                )
                depth_on_top = gr.Checkbox(
                    label="Depth on Top",
                    value=True,
                    visible=False
                )
            
            with gr.Row():
                blend_opacity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Blend Strength",
                    info="0 = Original only, 1 = Depth only",
                    visible=False
                )

        enable_blending.change(
            fn=lambda x: {
                make_base_bw: gr.update(visible=x),
                depth_on_top: gr.update(visible=x),
                blend_opacity: gr.update(visible=x)
            },
            inputs=[enable_blending],
            outputs=[make_base_bw, depth_on_top, blend_opacity]
        )

        with gr.Row():
            output_dir = gr.Textbox(
                label="Local Output Directory (Optional)", 
                placeholder="Leave empty to not save locally, or enter path to save files"
            )
            
        with gr.Row():
            upload_to_hf_toggle = gr.Checkbox(label="Upload to Hugging Face", value=True)
            repo_id_input = gr.Textbox(
                label="Hugging Face Repo ID", 
                placeholder="username/repo-name",
                interactive=True
            )
            
        with gr.Row():
            break_every_input = gr.Slider(
                minimum=50,
                maximum=200,
                value=100,
                step=10,
                label="Break Interval (for HF upload)"
            )
            
            resume_file = gr.Textbox(
                label="Resume File (Optional)", 
                placeholder="Leave empty for new uploads, or provide path to resume file"
            )
        
        process_button = gr.Button("Process Images", variant="primary")
        output = gr.Textbox(label="Output")
        
        # Live preview gallery
        gr.Markdown("### Live Processing Preview")
        live_preview = gr.Gallery(label="Processing Progress", columns=2, height=400)
        refresh_button = gr.Button("Refresh Preview")
        
    with gr.Tab("Preview"):
        with gr.Row():
            preview_folder = gr.Textbox(label="Folder Path", placeholder="Enter the path to preview images from")
            preview_model = gr.Dropdown(
                choices=["Small", "Base", "Large"],
                value="Small",
                label="Model Size"
            )
        
        with gr.Row():
            preview_invert = gr.Checkbox(label="Invert Depth Map", value=False)
            preview_colormap = gr.Dropdown(
                choices=colormap_list(),
                value="TURBO",
                label="Colormap Style"
            )
            
        with gr.Row():
            preview_blend_opacity = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.1,
                label="Preview Blend Opacity"
            )
            preview_make_bw = gr.Checkbox(
                label="Make Base Image Black & White",
                value=False
            )
            preview_depth_on_top = gr.Checkbox(
                label="Depth Map on Top",
                value=True
            )

        visualize_button = gr.Button("Generate Preview", variant="secondary")
        preview_output = gr.Gallery(label="Sample Depth Maps", columns=2, height=600)
    
    with gr.Tab("Help"):
        gr.Markdown("""
        ## Usage Instructions
        
        ### Generate Depth Maps Tab
        1. **Folder Path**: Enter the full path to the folder containing your images (PNG, JPG, JPEG)
        2. **Model Size**: 
           - Small: Fastest processing but lowest quality
           - Base: Good balance between speed and quality
           - Large: Best quality but slowest processing
        3. **Parallel Workers**: How many images to process simultaneously (only works with GPU)
        4. **Invert Depth Map**: Toggle to invert the depth values (far objects bright, near objects dark)
        5. **Colormap Style**: Choose from various color schemes for the depth visualization
        6. **Local Output Directory**: Path where you want to save processed images locally
        7. **Upload to Hugging Face**: Toggle whether to upload to Hugging Face Hub
        8. **HF Repo ID**: Your Hugging Face username and repository name (e.g., `username/dataset-name`)
        9. **Break Interval**: The script will take a 3-minute break after uploading this many files
        10. **Resume File**: If your upload was interrupted, you can provide the resume file path here
        
        ### Live Preview
        - During processing, a live preview will show the most recent processed images
        - Click "Refresh Preview" to update the display
        
        ### Preview Tab
        Quickly preview what the depth maps will look like without uploading anything.
        
        ### Important Notes
        - Processing is much faster with a GPU
        - If saving locally, original images and depth maps will be saved with _depth suffix
        - When uploading to Hugging Face, the script takes breaks to avoid rate limits
        """)
    
    # Define event handlers
    def toggle_hf_fields(upload_enabled):
        return {
            repo_id_input: gr.update(interactive=upload_enabled),
            break_every_input: gr.update(interactive=upload_enabled),
            resume_file: gr.update(interactive=upload_enabled)
        }
    
    # Connect interactive elements
    upload_to_hf_toggle.change(
        fn=toggle_hf_fields,
        inputs=upload_to_hf_toggle,
        outputs=[repo_id_input, break_every_input, resume_file]
    )
    
    # Connect buttons to functions
    process_button.click(
        fn=process_and_upload,
        inputs=[
            folder_input, model_dropdown, invert_depth, colormap_dropdown,
            output_dir, upload_to_hf_toggle, repo_id_input, 
            break_every_input, parallel_workers, resume_file,
            enable_blending, blend_opacity, make_base_bw, depth_on_top,
            use_colormap, reverse_colormap  # Add reverse_colormap
        ],
        outputs=output
    )
    
    refresh_button.click(
        fn=update_live_preview,
        inputs=[],
        outputs=live_preview
    )
    
    visualize_button.click(
        fn=visualize_process,
        inputs=[preview_folder, preview_model, preview_invert, preview_colormap,
                preview_blend_opacity, preview_make_bw, preview_depth_on_top],
        outputs=preview_output
    )
    
    # Set up the live preview - just initialize it
    demo.load(lambda: [], None, live_preview)
    
if __name__ == "__main__":
    demo.launch()
