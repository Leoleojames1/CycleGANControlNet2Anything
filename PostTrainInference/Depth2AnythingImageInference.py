import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import numpy as np
import cv2
from huggingface_hub import hf_hub_download

# Set up path for depth-anything
depth_anything_path = os.getenv('DEPTH_ANYTHING_V2_PATH')
if depth_anything_path is None:
    depth_anything_path = "./depth_anything_v2"  # Default path, modify as needed
    print(f"Environment variable DEPTH_ANYTHING_V2_PATH not set. Using default: {depth_anything_path}")
sys.path.append(depth_anything_path)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print("Successfully imported DepthAnythingV2")
except ImportError:
    print("Warning: Could not import DepthAnythingV2. Please check your DEPTH_ANYTHING_V2_PATH")

# Device selection with MPS support
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

###########################################
# Model Architecture
###########################################

# Generator architecture (simplified ResNet)
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
    os.makedirs("checkpoints", exist_ok=True)
    model_path = hf_hub_download(
        repo_id=model_info['repo_id'],
        filename=model_info['filename'],
        local_dir='checkpoints'
    )
    return model_path

def load_depth_model(encoder):
    """Load the specified depth model"""
    global current_depth_model, current_encoder
    if current_depth_model is None or current_encoder != encoder:
        try:
            model_path = download_model(encoder)
            current_depth_model = DepthAnythingV2(**model_configs[encoder])
            current_depth_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            current_depth_model = current_depth_model.to(DEVICE).eval()
            current_encoder = encoder
            print(f"Successfully loaded depth model: {encoder}")
        except Exception as e:
            print(f"Error loading depth model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    return current_depth_model

# Image preprocessing for CycleGAN
def preprocess_image(image_np):
    # Convert numpy array to PIL Image
    if isinstance(image_np, str):
        # If path is provided instead of numpy array
        image = Image.open(image_np).convert('RGB')
    else:
        # Ensure input image is RGB
        if len(image_np.shape) == 2:  # Grayscale
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[-1] == 4:  # RGBA
            image_np = image_np[..., :3]
        image = Image.fromarray(image_np.astype('uint8'), 'RGB')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Image postprocessing
def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy()
    return (tensor * 255).astype(np.uint8)

@torch.inference_mode()
def predict_depth(image, encoder):
    """Predict depth using the selected model"""
    model = load_depth_model(encoder)
    if model is None:
        return None
    
    # Convert to RGB if it's not already
    if isinstance(image, np.ndarray):
        if image.shape[-1] == 3:
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            # Check if image is BGR (from OpenCV)
            if image.shape[-1] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            # Handle grayscale or other formats
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # If already a PIL image or other format
        image_rgb = image
    
    try:
        depth = model.infer_image(image_rgb)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        return depth
    except Exception as e:
        print(f"Error predicting depth: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def apply_winter_colormap(depth_map):
    """Apply a winter-themed colormap to the depth map"""
    # Use COLORMAP_WINTER for blue to teal colors
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_WINTER)
    return depth_colored

def apply_colormap(depth_map, colormap_name):
    """Apply specified colormap to the depth map"""
    colormap_mapping = {
        "winter": cv2.COLORMAP_WINTER,
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "cividis": cv2.COLORMAP_CIVIDIS
    }
    
    colormap = colormap_mapping.get(colormap_name.lower(), cv2.COLORMAP_WINTER)
    depth_colored = cv2.applyColorMap(depth_map, colormap)
    return depth_colored

def blend_images(original, depth_colored, alpha=0.1):
    """
    Blend the original image on top of the colored depth map
    
    Parameters:
    - original: Original image (BGR format)
    - depth_colored: Colorized depth map (BGR format)
    - alpha: Blend strength of original image (0.0 = depth only, 1.0 = original only)
    
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

# Model loading for CycleGAN
def load_cyclegan_model(model_path):
    try:
        model = Generator()
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: {e}")
                # Try loading with strict=False
                model.load_state_dict(state_dict, strict=False)
                print("Loaded model with strict=False")
        else:
            print(f"Error: Model file not found at {model_path}")
            return None
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading CycleGAN model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Main processing function
@torch.inference_mode()
def process_image(input_image, processing_mode, depth_model="Small", 
                 colormap="winter", blend_alpha=0.1, use_default_paths=True,
                 depth2img_path=None, img2depth_path=None):
    """
    Process input image with various modes:
    - "direct_d2i": Use input depth map directly for Depth to Image conversion
    - "direct_i2d": Use input image directly for Image to Depth conversion
    - "img2depth2img": Process as Image -> Depth (with colormap) -> Image
    """
    if input_image is None:
        print("No input image provided")
        return None, "Error: No input image provided"
    
    try:
        # Default model paths
        if use_default_paths:
            depth2img_path = "../TrainNotebooksCycleGAN/TrainWithDepthDataset/checkpoints/depth2image/latest_net_G_A.pth"
            img2depth_path = "../TrainNotebooksCycleGAN/TrainWithDepthDataset/checkpoints/depth2image/latest_net_G_B.pth"
        
        # Ensure input image is RGB numpy array
        if len(input_image.shape) == 2:  # Grayscale
            input_image = np.stack([input_image] * 3, axis=-1)
        elif input_image.shape[-1] == 4:  # RGBA
            input_image = input_image[..., :3]
        
        # Convert to BGR for OpenCV functions
        input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        if processing_mode == "direct_d2i":
            # Direct Depth to Image mode - assume input is a depth map
            cyclegan_model = load_cyclegan_model(depth2img_path)
            if cyclegan_model is None:
                return None, f"Failed to load model from {depth2img_path}"
            
            # Preprocess
            input_tensor = preprocess_image(input_image)
            
            # Generate output
            output_tensor = cyclegan_model(input_tensor)
            
            # Postprocess
            output_image = postprocess_image(output_tensor)
            
            return output_image, "Direct Depth to Image conversion completed"
            
        elif processing_mode == "direct_i2d":
            # Direct Image to Depth mode
            cyclegan_model = load_cyclegan_model(img2depth_path)
            if cyclegan_model is None:
                return None, f"Failed to load model from {img2depth_path}"
            
            # Preprocess
            input_tensor = preprocess_image(input_image)
            
            # Generate output
            output_tensor = cyclegan_model(input_tensor)
            
            # Postprocess
            output_image = postprocess_image(output_tensor)
            
            return output_image, "Direct Image to Depth conversion completed"
            
        elif processing_mode == "img2depth2img":
            # Full pipeline: Image -> Depth (with colormap) -> Image
            # 1. Generate depth map using Depth Anything
            encoder = {v: k for k, v in encoder2name.items()}[depth_model]
            depth_map = predict_depth(input_image, encoder)
            if depth_map is None:
                return None, "Failed to predict depth map"
            
            # 2. Apply colormap and blend
            depth_colored = apply_colormap(depth_map, colormap)
            blended = blend_images(input_bgr, depth_colored, alpha=blend_alpha)
            
            # 3. Convert to RGB for CycleGAN
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            
            # 4. Apply CycleGAN (Depth to Image)
            cyclegan_model = load_cyclegan_model(depth2img_path)
            if cyclegan_model is None:
                # If CycleGAN fails, return the blended image as a fallback
                return blended_rgb, "CycleGAN model failed to load, returning depth visualization"
            
            # Preprocess
            input_tensor = preprocess_image(blended_rgb)
            
            # Generate output
            output_tensor = cyclegan_model(input_tensor)
            
            # Postprocess
            output_image = postprocess_image(output_tensor)
            
            # Prepare intermediate visualizations
            intermediate_results = {
                "depth_map": depth_map,
                "colorized_depth": cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB),
                "blended": blended_rgb
            }
            
            return output_image, "Image to Depth to Image conversion completed"
        
        else:
            return None, f"Unknown processing mode: {processing_mode}"
            
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# Custom CSS for a more attractive UI
custom_css = """
.gradio-container {
    background: linear-gradient(to bottom right, #1a2a6c, #b21f1f, #fdbb2d);
    color: white;
}
.gr-button-primary {
    background: #4082f0 !important;
    border: none !important;
}
.gr-button-secondary {
    border: 2px solid #4082f0 !important;
    color: #4082f0 !important;
}
.gr-input, .gr-select {
    border: 2px solid rgba(64, 130, 240, 0.5) !important;
    border-radius: 8px !important;
}
.gr-form {
    border-radius: 12px !important;
    background: rgba(0, 0, 0, 0.2) !important;
    padding: 20px !important;
}
.gr-panel {
    border-radius: 12px !important;
}
.gr-box {
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}
"""

# Update the Gradio interface
with gr.Blocks(title="Depth2Anything Enhanced UI", css=custom_css) as demo:
    gr.HTML("<h1 style='text-align: center; margin-bottom: 1rem; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>Depth2Anything Enhanced UI</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div style='background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 10px;'>")
            processing_mode = gr.Radio(
                choices=[
                    "img2depth2img", 
                    "direct_d2i", 
                    "direct_i2d"
                ],
                value="img2depth2img",
                label="Processing Mode",
                info="Choose how to process your image",
                container=True,
            )
            
            with gr.Group():
                gr.HTML("<h3 style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>Depth Estimation Settings</h3>")
                depth_model = gr.Dropdown(
                    choices=list(encoder2name.values()),
                    value="Small",
                    label="Depth Model Size",
                    info="Smaller models are faster, larger are more accurate",
                    visible=True
                )
                
                colormap = gr.Dropdown(
                    choices=["winter", "jet", "hot", "rainbow", "viridis", "plasma", "inferno", "magma", "cividis"],
                    value="winter",
                    label="Depth Colormap",
                    info="Color scheme for visualizing depth",
                    visible=True
                )
                
                blend_alpha = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Original Image Overlay",
                    info="0 = depth only, 1 = original image only",
                    visible=True
                )
            
            use_default_paths = gr.Checkbox(
                value=True,
                label="Use Default Model Paths",
                info="Uncheck to specify custom CycleGAN model paths"
            )
            
            with gr.Group(visible=False) as custom_paths_group:
                depth2img_path = gr.Textbox(
                    label="Depth to Image Model Path",
                    value="../TrainNotebooksCycleGAN/TrainWithDepthDataset/checkpoints/depth2image/latest_net_G_A.pth",
                    info="Path to the CycleGAN model for converting depth maps to images"
                )
                
                img2depth_path = gr.Textbox(
                    label="Image to Depth Model Path",
                    value="../TrainNotebooksCycleGAN/TrainWithDepthDataset/checkpoints/depth2image/latest_net_G_B.pth",
                    info="Path to the CycleGAN model for converting images to depth maps"
                )
            
            transform_btn = gr.Button("Transform", variant="primary", size="lg")
            gr.HTML("</div>")
            
        with gr.Column(scale=2):
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    height=400,
                    width=400,
                    container=True,
                    elem_id="input-image"
                )
                output_image = gr.Image(
                    label="Generated Output",
                    type="numpy",
                    height=400,
                    width=400,
                    container=True,
                    elem_id="output-image"
                )
                
            with gr.Row():
                error_output = gr.Textbox(
                    label="Status",
                    value="Ready. Upload an image and click Transform.",
                    interactive=False
                )
    
    with gr.Row():
        # Tabs for intermediate results when using img2depth2img mode
        with gr.Tabs():
            with gr.TabItem("Instructions"):
                gr.Markdown("""
                ### Processing Modes:
                
                1. **Image to Depth to Image (img2depth2img)**
                   - Takes your input image
                   - Uses Depth Anything to predict a depth map
                   - Applies colormap and blending
                   - Converts depth visualization back to image with CycleGAN
                
                2. **Direct Depth to Image (direct_d2i)**
                   - Use this if your input is already a depth map
                   - Directly applies CycleGAN to convert depth to image
                
                3. **Direct Image to Depth (direct_i2d)**
                   - Converts your image to a depth map using CycleGAN
                   - Does not use Depth Anything
                
                ### Tips:
                - For best results with Depth Anything, use high-quality, well-lit images
                - The Winter colormap gives the best results with CycleGAN
                - Try different blend values to adjust original image visibility in depth view
                - Larger depth models provide more accurate depth maps but are slower
                """)
            
            with gr.TabItem("Examples"):
                gr.Examples(
                    examples=[
                        ["examples/landscape.jpg", "img2depth2img", "Small", "winter", 0.1],
                        ["examples/portrait.jpg", "img2depth2img", "Base", "plasma", 0.2],
                        ["examples/depth_sample.png", "direct_d2i", "Small", "winter", 0.1],
                    ],
                    inputs=[input_image, processing_mode, depth_model, colormap, blend_alpha],
                    outputs=[output_image, error_output],
                    fn=lambda *args: process_image(*args),
                    cache_examples=True,
                )
    
    def toggle_custom_paths(use_default):
        return gr.Group.update(visible=not use_default)
    
    def update_ui_based_on_mode(mode):
        if mode == "img2depth2img":
            return [
                gr.Dropdown.update(visible=True),  # depth_model
                gr.Dropdown.update(visible=True),  # colormap
                gr.Slider.update(visible=True),    # blend_alpha
            ]
        else:
            return [
                gr.Dropdown.update(visible=False),  # depth_model
                gr.Dropdown.update(visible=False),  # colormap
                gr.Slider.update(visible=False),    # blend_alpha
            ]
    
    # Connect components and set up event handlers
    use_default_paths.change(
        fn=toggle_custom_paths,
        inputs=use_default_paths,
        outputs=custom_paths_group
    )
    
    processing_mode.change(
        fn=update_ui_based_on_mode,
        inputs=processing_mode,
        outputs=[depth_model, colormap, blend_alpha]
    )
    
    transform_btn.click(
        fn=process_image,
        inputs=[
            input_image, 
            processing_mode, 
            depth_model, 
            colormap, 
            blend_alpha, 
            use_default_paths,
            depth2img_path,
            img2depth_path
        ],
        outputs=[output_image, error_output]
    )

if __name__ == "__main__":
    # Make sure checkpoints directory exists
    os.makedirs("checkpoints/depth2image", exist_ok=True)
    
    # Launch with custom server configuration
    demo.queue(max_size=5).launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Set specific port
        show_error=True,  # Show detailed errors
        debug=True  # Enable debug mode
    )