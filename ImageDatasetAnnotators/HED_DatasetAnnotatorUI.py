#!/usr/bin/env python

import os
import sys
import shutil
import gradio as gr
import torch
import numpy as np
import PIL.Image
from PIL import Image
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import base64
import io

# Import the model using environment variable
# Import the model using command-line argument or environment variable or default path
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='HED Edge Detection Dataset Annotator')
parser.add_argument('--hed_path', type=str, default=None, help='Path to HED model directory')
args, unknown = parser.parse_known_args()

# Determine HED path in this order: command-line arg, environment variable, default paths
hed_path = args.hed_path
if hed_path is None:
    hed_path = os.getenv('HED_PATH')
if hed_path is None:
    # Try some default paths
    possible_paths = [
        '.',  # Current directory
        'M:\\PHOTO_HDD_AUTUMN_GAN\\pytorch-hed',  # Your specific path
        os.path.join(os.path.dirname(__file__), '..', 'pytorch-hed'),  # One level up
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pytorch-hed')  # Absolute path
    ]
    for path in possible_paths:
        if os.path.exists(path):
            hed_path = path
            print(f"Using default HED path: {hed_path}")
            break

if hed_path is None or not os.path.exists(hed_path):
    raise ValueError(f"HED model path not found. Please provide a valid path using --hed_path argument or HED_PATH environment variable.")

sys.path.append(hed_path)

try:
    from run import Network, estimate
    print(f"Successfully imported HED model from {hed_path}")
except ImportError as e:
    raise ImportError(f"Could not import HED model from {hed_path}: {e}")

# Global variables
processed_images = []
output_dir = None

# Custom CSS with simplified styling and taller galleries
custom_css = """
:root {
    --slate-color: #3F4756;
    --mustard-color: #E5B22B;
    --dark-slate: #2C3241;
    --light-slate: #616C7C;
    --light-yellow: #F8E9B7;
    --warning-red: #D64045;
}

body {
    background-color: var(--slate-color) !important;
    color: white !important;
}

.gradio-container {
    max-width: 95% !important;
}

/* Gallery styling for taller display */
.gallery-container .svelte-1p8za3 {
    height: 600px !important;
}

.construction-header {
    background-color: var(--mustard-color);
    color: black;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 5px solid #333;
}

button.primary {
    background-color: var(--mustard-color) !important;
    color: black !important;
    font-weight: bold !important;
    border: 2px solid black !important;
}

button {
    background-color: var(--dark-slate) !important;
    border: 2px solid var(--mustard-color) !important;
    color: white !important;
}

.container-box {
    background-color: var(--dark-slate);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    border: 2px solid var(--mustard-color);
}

.caution-divider {
    height: 15px;
    background: repeating-linear-gradient(
        45deg,
        black,
        black 10px,
        var(--mustard-color) 10px,
        var(--mustard-color) 20px
    );
    margin: 20px 0;
    border-radius: 2px;
}

.info-box {
    background-color: var(--light-slate);
    color: white;
    padding: 15px;
    border-radius: 5px;
    margin: 15px 0;
    border-left: 5px solid var(--mustard-color);
}

.warning-box {
    background-color: var(--warning-red);
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    border: 2px solid black;
}

.footer {
    background-color: var(--mustard-color);
    color: black;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
    border: 3px solid #333;
}

/* Construction icon styling */
.construction-icon {
    display: inline-block;
    font-size: 24px;
    margin-right: 10px;
    vertical-align: middle;
}
"""

def create_output_dir(input_dir):
    """Create an output directory based on the input directory name"""
    input_path = Path(input_dir)
    output_path = input_path.parent / f"{input_path.name}_hed_output"
    os.makedirs(output_path, exist_ok=True)
    return str(output_path)

def process_image(input_path):
    """Process a single image with HED"""
    try:
        # Read the image
        img = PIL.Image.open(input_path)
        img = img.convert("RGB")
        
        # Resize image to 480x320 if needed (as required by the model)
        orig_size = img.size
        img_resized = img.resize((480, 320), PIL.Image.LANCZOS)
        
        # Convert to tensor
        img_np = np.array(img_resized)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        ten_input = torch.FloatTensor(np.ascontiguousarray(img_np))
        
        # Move to GPU if available
        if torch.cuda.is_available():
            ten_input = ten_input.cuda()
        
        # Process with the model
        ten_output = estimate(ten_input)
        
        # Convert back to PIL image
        output_np = (ten_output.clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)
        output_img = PIL.Image.fromarray(output_np)
        
        # Resize back to original dimensions if needed
        if orig_size != (480, 320):
            output_img = output_img.resize(orig_size, PIL.Image.LANCZOS)
            
        return output_img
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return None

def process_folder(input_dir, output_folder=None, progress=gr.Progress()):
    """Process all images in a folder"""
    global processed_images, output_dir
    
    # Remember output directory for other functions
    if not output_folder or output_folder.strip() == "":
        output_dir = create_output_dir(input_dir)
    else:
        output_dir = output_folder
    
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = [
        f for f in os.listdir(input_dir) 
        if os.path.isfile(os.path.join(input_dir, f)) and 
        any(f.lower().endswith(ext) for ext in valid_extensions)
    ]
    
    if not image_files:
        return [], [], f"⚠️ No image files found in {input_dir}"
    
    # Initialize progress tracking
    progress(0, desc="🚧 STARTING EDGE DETECTION PROCESSING 🚧")
    processed_images = []
    
    # Process each image
    for i, img_file in enumerate(image_files):
        input_path = os.path.join(input_dir, img_file)
        
        # Update progress
        progress((i / len(image_files)), desc=f"🔄 Processing image {i+1}/{len(image_files)}: {img_file}")
        
        # Process the image
        output_img = process_image(input_path)
        if output_img is None:
            continue
            
        # Save input and output images to the dataset folder
        input_copy_path = os.path.join(dataset_dir, f"input_{i:04d}{os.path.splitext(img_file)[1]}")
        output_path = os.path.join(dataset_dir, f"output_{i:04d}{os.path.splitext(img_file)[1]}")
        
        # Copy original to dataset
        shutil.copy(input_path, input_copy_path)
        
        # Save edge map to dataset
        output_img.save(output_path)
        
        # For preview, use original images (better quality)
        input_thumb = Image.open(input_path)
        
        # Add to processed images
        processed_images.append((input_thumb, output_img, os.path.basename(input_path)))
    
    # Final progress update
    progress(1.0, desc="✅ PROCESSING COMPLETE")
    
    # Return thumbnails for gallery display
    input_images = [pair[0] for pair in processed_images[:8]]  # First 8 input images
    output_images = [pair[1] for pair in processed_images[:8]]  # First 8 output images
    
    completion_message = f"✅ CONSTRUCTION COMPLETE! \n🏗️ Processed {len(processed_images)} images \n📊 Dataset created in: {dataset_dir}\n\n(Showing first 8 images in gallery, all images saved to dataset folder)"
    
    return input_images, output_images, completion_message

def open_folder(folder_path):
    """Open a folder in file explorer"""
    if folder_path and os.path.exists(folder_path):
        if sys.platform == 'win32':
            os.startfile(folder_path)
        elif sys.platform == 'darwin':  # macOS
            os.system(f'open "{folder_path}"')
        else:  # Linux
            os.system(f'xdg-open "{folder_path}"')
        return f"🔍 Opened folder: {folder_path}"
    return "⚠️ Folder doesn't exist"

def browse_folder():
    """Open a folder browser dialog and return the selected path"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Make sure it appears on top
    folder_path = filedialog.askdirectory()
    return folder_path if folder_path else None

def view_more_images():
    """Open the output folder to view all processed images"""
    global output_dir
    if output_dir and os.path.exists(output_dir):
        return open_folder(output_dir)
    return "⚠️ No output folder available. Process images first."

# Initialize the model
print("Initializing HED model...")
model = Network()
if torch.cuda.is_available():
    model = model.cuda()
model.train(False)
print("Model loaded!")

# Create Gradio interface
with gr.Blocks(title="HED Edge Detection - Construction Zone", css=custom_css) as app:
    # Custom Header
    gr.HTML("""
    <div class="construction-header">
        <h1>🏗️ EDGE DETECTION CONSTRUCTION ZONE 🏗️</h1>
        <p>Build perfect edge maps from your images with HED technology</p>
    </div>
    """)
    
    # Main content
    with gr.Row():
        with gr.Column(scale=1):
            # Container for controls
            gr.HTML('<div class="container-box">')
            gr.HTML("<h3>🔧 CONTROL PANEL 🔧</h3>")
            
            # Environment Variable Info
            gr.HTML("""
            <div class="info-box">
                <h4>🔧 ENVIRONMENT SETUP</h4>
                <p>This application requires the following environment variable:</p>
                <ul>
                    <li><b>HED_PATH:</b> Path to the HED model directory</li>
                </ul>
                <p>Example: <code>export HED_PATH=/path/to/hed</code></p>
                <p>Current HED_PATH: <code>""" + hed_path + """</code></p>
            </div>
            """)
            
            # Caution divider
            gr.HTML('<div class="caution-divider"></div>')
            
            # Input components
            gr.HTML('<span class="construction-icon">📁</span><b>PROJECT MATERIALS</b>')
            input_folder = gr.Textbox(label="Input Folder", placeholder="Select your input folder")
            browse_input_button = gr.Button("📂 Browse Input Folder")
            
            gr.HTML('<div class="caution-divider"></div>')
            
            gr.HTML('<span class="construction-icon">🔨</span><b>CONSTRUCTION OUTPUT</b>')
            output_folder = gr.Textbox(label="Output Folder (optional)", placeholder="Select output destination (or leave empty)")
            browse_output_button = gr.Button("📂 Browse Output Folder")
            
            gr.HTML('<div class="caution-divider"></div>')
            
            # Action buttons
            gr.HTML('<span class="construction-icon">⚡</span><b>OPERATIONS</b>')
            process_button = gr.Button("🚧 START CONSTRUCTION 🚧", elem_classes=["primary"])
            
            with gr.Row():
                open_input_button = gr.Button("🔍 View Input Site")
                open_output_button = gr.Button("🔍 View Output Site")
            
            # Add view more button
            view_more_button = gr.Button("🔍 View All Processed Images")
            
            # Status textbox
            status_text = gr.Textbox(
                label="CONSTRUCTION STATUS", 
                interactive=False,
                value="🚧 Ready to build! Select folders and start construction.",
                lines=5
            )
            
            gr.HTML('</div>') # Close container-box
            
            # Info box
            gr.HTML("""
            <div class="info-box">
                <h4>🔔 SITE INFORMATION</h4>
                <ul>
                    <li>All edge maps will be saved directly to the dataset folder</li>
                    <li>Images are temporarily resized to 480x320 during processing</li>
                    <li>Input/output pairs are saved with matching indices for training</li>
                    <li>Gallery shows first 8 images - use "View All Processed Images" to see all results</li>
                </ul>
            </div>
            """)
            
            # Warning box
            gr.HTML("""
            <div class="warning-box">
                <h4>⚠️ SAFETY FIRST!</h4>
                <p>This operation requires CUDA to run efficiently. CPU processing will be extremely slow.</p>
            </div>
            """)
            
        with gr.Column(scale=2):
            # Construction site viewer with separate galleries
            gr.HTML("<h3>🏗️ CONSTRUCTION SITE VIEWER 🏗️</h3>")
            
            with gr.Row():
                with gr.Column():
                    # Input images with taller height
                    input_gallery = gr.Gallery(
                        label="INPUT IMAGES", 
                        show_label=True,
                        height=600,  # Increased from 400 to 600
                        object_fit="contain",
                        elem_classes=["gallery-container"]
                    )
                
                with gr.Column():
                    # Output images (HED) with taller height
                    output_gallery = gr.Gallery(
                        label="HED EDGE MAPS", 
                        show_label=True,
                        height=600,  # Increased from 400 to 600
                        object_fit="contain",
                        elem_classes=["gallery-container"]
                    )
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <p>🏗️ HOLISTICALLY-NESTED EDGE DETECTION (HED) CONSTRUCTION EQUIPMENT 🏗️</p>
        <p>Building better edges since 2015 | Hard Hat Area | Authorized Personnel Only</p>
    </div>
    """)
    
    # Event handlers for folder selection
    browse_input_button.click(
        fn=browse_folder, 
        outputs=input_folder
    )
    
    browse_output_button.click(
        fn=browse_folder, 
        outputs=output_folder
    )
    
    # Process button handler
    process_button.click(
        fn=process_folder,
        inputs=[input_folder, output_folder],
        outputs=[input_gallery, output_gallery, status_text]
    )
    
    # Open folder buttons
    open_input_button.click(
        fn=open_folder,
        inputs=input_folder,
        outputs=status_text
    )
    
    open_output_button.click(
        fn=open_folder,
        inputs=output_folder,
        outputs=status_text
    )
    
    # View more images button
    view_more_button.click(
        fn=view_more_images,
        outputs=status_text
    )

if __name__ == "__main__":
    # Print environment variable information
    print(f"HED_PATH: {hed_path}")
    
    # Make sure cuda is available
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA is not available. The model will run on CPU and be EXTREMELY slow.")
    else:
        print("CUDA is available. Using GPU for processing.")
    
    # Launch the app
    app.launch(share=False)