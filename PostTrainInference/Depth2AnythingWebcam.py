import gradio as gr
import cv2
import numpy as np
import torch
import sys
import os
import pyvirtualcam
from pyvirtualcam import PixelFormat
from huggingface_hub import hf_hub_download
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Path configurations
depth_anything_path = os.getenv('DEPTH_ANYTHING_V2_PATH')
if depth_anything_path is None:
    raise ValueError("Environment variable DEPTH_ANYTHING_V2_PATH is not set. Please set it to the path of Depth-Anything-V2")
sys.path.append(depth_anything_path)

from depth_anything_v2.dpt import DepthAnythingV2

# Device selection with MPS support
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

###########################################
# CycleGAN Generator Architecture
###########################################

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

###########################################
# Depth Anything Model Functions
###########################################

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
current_depth_model = None
current_encoder = None
current_cyclegan_model = None

def download_model(encoder):
    """Download the specified model from HuggingFace Hub"""
    model_info = MODEL_INFO[encoder]
    model_path = hf_hub_download(
        repo_id=model_info['repo_id'],
        filename=model_info['filename'],
        local_dir='checkpoints'
    )
    return model_path

def load_depth_model(encoder):
    """Load the specified depth model"""
    global current_depth_model, current_encoder
    if current_encoder != encoder:
        model_path = download_model(encoder)
        current_depth_model = DepthAnythingV2(**model_configs[encoder])
        current_depth_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        current_depth_model = current_depth_model.to(DEVICE).eval()
        current_encoder = encoder
    return current_depth_model

def load_cyclegan_model(model_path):
    """Load the CycleGAN model"""
    global current_cyclegan_model
    if current_cyclegan_model is None:
        model = Generator()
        if os.path.exists(model_path):
            print(f"Loading CycleGAN model from {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: {e}")
                # Try loading with strict=False
                model.load_state_dict(state_dict, strict=False)
                print("Loaded model with strict=False")
        else:
            print(f"Error: CycleGAN model file not found at {model_path}")
            return None
        model.eval()
        current_cyclegan_model = model.to(DEVICE)
    return current_cyclegan_model

@torch.inference_mode()
def predict_depth(image, encoder):
    """Predict depth using the selected model"""
    model = load_depth_model(encoder)
    depth = model.infer_image(image)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return depth

def apply_winter_colormap(depth_map):
    """Apply a winter-themed colormap to the depth map"""
    # Use COLORMAP_WINTER for blue to teal colors
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_WINTER)
    return depth_colored

def blend_images(original, depth_colored, alpha=0.1):
    """
    Blend the original image on top of the colored depth map
    
    Parameters:
    - original: Original webcam frame (BGR format)
    - depth_colored: Colorized depth map (BGR format)
    - alpha: Blend strength of original webcam (0.0 = depth only, 1.0 = original only)
    
    Returns:
    - Blended image where depth map is the base layer and original is overlaid with transparency
    """
    # Make sure both images have the same dimensions
    if original.shape != depth_colored.shape:
        depth_colored = cv2.resize(depth_colored, (original.shape[1], original.shape[0]))
    
    # Start with depth map at 100% opacity as base
    # Then add original image on top with specified alpha transparency
    result = cv2.addWeighted(depth_colored, 1.0, original, alpha, 0)
    
    return result

def preprocess_for_cyclegan(image, original_size=None):
    """Preprocess image for CycleGAN input"""
    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(image)
    
    # Save original size if provided
    if original_size is None:
        original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Process image
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    return input_tensor, original_size

def postprocess_from_cyclegan(tensor, original_size):
    """Convert CycleGAN output tensor to numpy image with original dimensions"""
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy()
    # Convert to uint8
    image = (tensor * 255).astype(np.uint8)
    # Resize back to original dimensions
    if image.shape[0] != original_size[1] or image.shape[1] != original_size[0]:
        image = cv2.resize(image, original_size)
    return image

@torch.inference_mode()
def apply_cyclegan(image, direction):
    """Apply CycleGAN transformation to the image"""
    if direction == "Depth to Image":
        model_path = "../TrainNotebooksCycleGAN/TrainWithDepthDataset/checkpoints/depth2image/latest_net_G_A.pth"
    else:
        model_path = "../TrainNotebooksCycleGAN/TrainWithDepthDataset/checkpoints/depth2image/latest_net_G_B.pth"
    
    model = load_cyclegan_model(model_path)
    if model is None:
        return None
    
    # Save original dimensions
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Preprocess
    input_tensor, _ = preprocess_for_cyclegan(image, original_size)
    
    # Generate output
    output_tensor = model(input_tensor)
    
    # Postprocess with original size
    output_image = postprocess_from_cyclegan(output_tensor, original_size)
    
    return output_image

def process_webcam_with_depth_and_cyclegan(encoder, blend_alpha, cyclegan_direction, enable_cyclegan=True):
    """Process webcam with depth, blend, and optionally apply CycleGAN"""
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Read a test frame to get the actual dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam")
        return
    
    # Get the actual frame dimensions
    frame_height, frame_width = test_frame.shape[:2]
    print(f"Webcam frame dimensions: {frame_width}x{frame_height}")
    
    # Ensure checkpoints directory exists
    os.makedirs("checkpoints/depth2image", exist_ok=True)
    
    # Create a preview window
    preview_window = "Depth Winter + CycleGAN Preview"
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    
    try:
        # Initialize virtual camera with exact frame dimensions
        with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=30, fmt=PixelFormat.BGR, backend='obs') as cam:
            print(f'Using virtual camera: {cam.device}')
            print(f'Virtual camera dimensions: {cam.width}x{cam.height}')
            
            frame_count = 0
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Print dimensions occasionally for debugging
                if frame_count % 100 == 0:
                    print(f"Frame {frame_count} dimensions: {frame.shape}")
                frame_count += 1
                
                # Convert BGR to RGB for depth prediction
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predict depth
                depth_map = predict_depth(frame_rgb, encoder)
                
                # Apply winter colormap
                depth_colored = apply_winter_colormap(depth_map)
                
                # Blend with original
                blended = blend_images(frame, depth_colored, alpha=blend_alpha)
                
                # Apply CycleGAN if enabled
                if enable_cyclegan:
                    if cyclegan_direction == "Image to Depth":
                        # For Image to Depth, use raw webcam feed (not blended)
                        input_for_gan = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        # For Depth to Image, use the blended result
                        input_for_gan = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                    
                    cyclegan_output = apply_cyclegan(input_for_gan, cyclegan_direction)
                    
                    if cyclegan_output is not None:
                        # Convert RGB back to BGR for virtual cam
                        output = cv2.cvtColor(cyclegan_output, cv2.COLOR_RGB2BGR)
                    else:
                        output = blended
                else:
                    output = blended
                
                # Ensure output has the exact dimensions expected by the virtual camera
                if output.shape[0] != frame_height or output.shape[1] != frame_width:
                    print(f"Resizing output from {output.shape[1]}x{output.shape[0]} to {frame_width}x{frame_height}")
                    output = cv2.resize(output, (frame_width, frame_height))
                
                # Show preview
                cv2.imshow(preview_window, output)
                
                # Send to virtual camera
                try:
                    cam.send(output)
                    cam.sleep_until_next_frame()
                except Exception as e:
                    print(f"Error sending to virtual camera: {e}")
                    print(f"Output shape: {output.shape}, Expected: {frame_height}x{frame_width}x3")
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except Exception as e:
        print(f"Error in webcam processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

###########################################
# Gradio Interface
###########################################

with gr.Blocks(title="Depth Anything with CycleGAN") as demo:
    gr.Markdown("# Depth Anything V2 with Winter Colormap + CycleGAN")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=list(encoder2name.values()),
                value="Small",
                label="Select Depth Model Size"
            )
            
            blend_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.1,  # Set default to 0.1 (10% webcam opacity)
                step=0.1,
                label="Webcam Overlay Opacity (0 = depth only, 1 = full webcam overlay)"
            )
            
            cyclegan_toggle = gr.Checkbox(
                value=True,
                label="Enable CycleGAN Transformation"
            )
            
            cyclegan_direction = gr.Radio(
                choices=["Depth to Image", "Image to Depth"],
                value="Depth to Image",
                label="CycleGAN Direction"
            )
            
            start_button = gr.Button("Start Processing", variant="primary")
        
        with gr.Column():
            output_status = gr.Textbox(
                label="Status",
                value="Ready to start...",
                interactive=False
            )
    
    # Instructions
    gr.Markdown("""
    ### Instructions:
    1. Select the depth model size (smaller models are faster but less accurate)
    2. Adjust the blend strength between the original webcam feed and the winter-colored depth map
    3. Enable/disable CycleGAN transformation
    4. Select the CycleGAN conversion direction
    5. Click "Start Processing" to begin the virtual camera feed
    6. A preview window will open - press 'q' in that window to stop processing
    
    **Note:** You'll need to have pyvirtualcam installed and a virtual camera device 
    (like OBS Virtual Camera) configured on your system.
    """)
    
    def start_processing(model_name, blend_alpha, enable_cyclegan, cyclegan_dir):
        encoder = {v: k for k, v in encoder2name.items()}[model_name]
        try:
            process_webcam_with_depth_and_cyclegan(
                encoder, 
                blend_alpha, 
                cyclegan_dir,
                enable_cyclegan
            )
            return "Processing completed. (If this message appears immediately, check for errors in the console)"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    start_button.click(
        fn=start_processing,
        inputs=[model_dropdown, blend_slider, cyclegan_toggle, cyclegan_direction],
        outputs=output_status
    )

if __name__ == "__main__":
    demo.launch()