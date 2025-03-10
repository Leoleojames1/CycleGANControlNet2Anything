import gradio as gr
import cv2
import numpy as np
import torch
import sys
import os
import pyvirtualcam
from pyvirtualcam import PixelFormat
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Import HED model using environment variable
hed_path = os.getenv('HED_PATH')
if hed_path is None:
    raise ValueError("Environment variable HED_PATH is not set. Please set it to the path of your HED model.")
sys.path.append(hed_path)

try:
    from run import Network, estimate
    print(f"Successfully imported HED model from {hed_path}")
except ImportError as e:
    raise ImportError(f"Could not import HED model from {hed_path}: {e}")

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

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Global variables for model management
current_hed_model = None
current_cyclegan_model = None

# Get CycleGAN path from environment variable or use default
CYCLEGAN_PATH = os.getenv('CYCLEGAN_PATH', '../TrainNotebooksCycleGAN/TrainWithHEDDataset/pytorch-CycleGAN-and-pix2pix')
print(f"Using CycleGAN path: {CYCLEGAN_PATH}")

###########################################
# HED Model Functions
###########################################

def prepare_edge_map_for_display(edge_map):
    """Convert grayscale edge map to 3-channel format for display without changing colors"""
    # Simply create a 3-channel image from the grayscale edge map
    edge_display = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    return edge_display

def load_hed_model():
    """Load the HED model"""
    global current_hed_model
    if current_hed_model is None:
        print("Initializing HED model...")
        current_hed_model = Network()
        if torch.cuda.is_available():
            current_hed_model = current_hed_model.cuda()
        current_hed_model.train(False)
        print("HED model loaded!")
    return current_hed_model

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
def predict_edges(image):
    """Predict edges using HED model"""
    load_hed_model()
    
    # Convert image to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    # Save original size
    original_size = (image.shape[1], image.shape[0])
    
    # Resize for HED model (which expects 480x320)
    img_resized = cv2.resize(image, (480, 320), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to tensor for HED
    img_np = np.array(img_resized)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    ten_input = torch.FloatTensor(np.ascontiguousarray(img_np))
    
    # Move to GPU if available
    if torch.cuda.is_available():
        ten_input = ten_input.cuda()
    
    # Process with HED model
    ten_output = estimate(ten_input)
    
    # Convert back to numpy
    edge_map = (ten_output.clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)
    
    # Resize back to original size
    edge_map = cv2.resize(edge_map, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    return edge_map

def capture_desktop():
    """Capture the desktop screen"""
    try:
        # Check if we can use pyautogui for screen capture
        import pyautogui
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        # Convert RGB to BGR (OpenCV format)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame
    except ImportError:
        print("pyautogui not installed. Trying mss for screen capture...")
        try:
            # Try using mss as an alternative
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                # Convert to numpy array
                frame = np.array(screenshot)
                # Remove alpha channel
                frame = frame[:, :, :3]
                return True, frame
        except ImportError:
            print("Neither pyautogui nor mss are installed. Cannot capture desktop.")
            return False, None
        except Exception as e:
            print(f"Error capturing desktop with mss: {e}")
            return False, None
    except Exception as e:
        print(f"Error capturing desktop with pyautogui: {e}")
        return False, None

def blend_images(original, edge_colored, alpha=0.1):
    """
    Blend the original image on top of the colored edge map
    
    Parameters:
    - original: Original webcam frame (BGR format)
    - edge_colored: Colorized edge map (BGR format)
    - alpha: Blend strength of original webcam (0.0 = edges only, 1.0 = original only)
    
    Returns:
    - Blended image where edge map is the base layer and original is overlaid with transparency
    """
    # Make sure both images have the same dimensions
    if original.shape != edge_colored.shape:
        edge_colored = cv2.resize(edge_colored, (original.shape[1], original.shape[0]))
    
    # Start with edge map at 100% opacity as base
    # Then add original image on top with specified alpha transparency
    result = cv2.addWeighted(edge_colored, 1.0, original, alpha, 0)
    
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
    if direction == "Edges to Image":
        model_path = os.path.join(CYCLEGAN_PATH, "checkpoints/hed2image/latest_net_G_A.pth")
    else:
        model_path = os.path.join(CYCLEGAN_PATH, "checkpoints/hed2image/latest_net_G_B.pth")
    
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

def process_webcam_with_hed_and_cyclegan(blend_alpha, cyclegan_direction, enable_cyclegan=True, use_desktop=False):
    """Process webcam/desktop with HED, blend, and optionally apply CycleGAN"""
    if use_desktop:
        # Check if we can capture the desktop
        can_capture, test_frame = capture_desktop()
        if not can_capture:
            print("Error: Could not capture desktop. Make sure pyautogui or mss is installed.")
            return "Error: Could not capture desktop. Please install pyautogui or mss."
        print("Successfully initialized desktop capture mode.")
    else:
        # Open the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return "Error: Could not open webcam. Please check your camera connection."
        
        # Read a test frame to get the actual dimensions
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            return "Error: Could not read from webcam."
    
    # Get the frame dimensions
    frame_height, frame_width = test_frame.shape[:2]
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    
    # Ensure checkpoints directory exists
    os.makedirs(os.path.join(CYCLEGAN_PATH, "checkpoints/hed2image"), exist_ok=True)
    
    # Create a preview window
    preview_window = "Quantum Edge Detection Studio"
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    
    try:
        # Initialize virtual camera with exact frame dimensions
        try:
            cam = pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=30, fmt=PixelFormat.BGR, backend='obs')
            print(f'Using virtual camera: {cam.device}')
            print(f'Virtual camera dimensions: {cam.width}x{cam.height}')
        except Exception as e:
            print(f"Could not initialize virtual camera: {e}")
            return "Error: Could not initialize virtual camera. Check if OBS Virtual Camera is installed."
        
        frame_count = 0
        while True:
            # Capture frame (from webcam or desktop)
            if use_desktop:
                ret, frame = capture_desktop()
                if not ret:
                    print("Error capturing desktop")
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Error capturing webcam frame")
                    break
            
            # Print dimensions occasionally for debugging
            if frame_count % 100 == 0:
                print(f"Frame {frame_count} dimensions: {frame.shape}")
            frame_count += 1
            
            # Convert BGR to RGB for edge prediction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict edges
            edge_map = predict_edges(frame_rgb)
            
            # Prepare edge map for display (3-channel grayscale)
            edge_colored = prepare_edge_map_for_display(edge_map)
            
            # Blend with original
            blended = blend_images(frame, edge_colored, alpha=blend_alpha)
            
            # Apply CycleGAN if enabled
            if enable_cyclegan:
                if cyclegan_direction == "Image to Edges":
                    # For Image to Edges, use raw feed (not blended)
                    input_for_gan = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # For Edges to Image, use the blended result
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
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return f"Error in processing: {str(e)}"
    
    finally:
        # Clean up
        if not use_desktop and 'cap' in locals():
            cap.release()
        if 'cam' in locals():
            cam.close()
        cv2.destroyAllWindows()
        
    return "Processing completed successfully."

###########################################
# Gradio Interface - Quantum-inspired soft red aesthetic
###########################################

# Custom CSS for quantum-inspired soft red aesthetic
custom_css = """
:root {
    --quantum-red: #FF6B6B;
    --soft-red: #FFABAB;
    --pastel-red: #FFD2D2;
    --darker-red: #D83A3A;
    --soft-black: #333333;
    --pure-white: #FFFFFF;
    --off-white: #F9F9F9;
    --light-gray: #EAEAEA;
}

body {
    background-color: var(--off-white) !important;
    color: var(--soft-black) !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}

.gradio-container {
    max-width: 95% !important;
}

.app-header {
    background: linear-gradient(135deg, var(--quantum-red) 0%, var(--darker-red) 100%);
    color: var(--pure-white);
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 24px;
    border: none;
    box-shadow: 0 4px 15px rgba(216, 58, 58, 0.15);
}

button.primary {
    background: linear-gradient(to right, var(--quantum-red), var(--darker-red)) !important;
    color: var(--pure-white) !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px rgba(216, 58, 58, 0.1) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(216, 58, 58, 0.2) !important;
}

button {
    background-color: var(--pure-white) !important;
    border: 1px solid var(--soft-red) !important;
    color: var(--soft-black) !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}

button:hover {
    background-color: var(--pastel-red) !important;
    color: var(--soft-black) !important;
}

.container-box {
    background-color: var(--pure-white);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    border: none;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.info-box {
    background-color: var(--pastel-red);
    color: var(--soft-black);
    padding: 18px;
    border-radius: 8px;
    margin: 18px 0;
    border-left: 5px solid var(--quantum-red);
    box-shadow: 0 2px 8px rgba(216, 58, 58, 0.08);
}

.divider {
    height: 3px;
    background: linear-gradient(to right, var(--pastel-red), var(--quantum-red), var(--pastel-red));
    margin: 24px 0;
    border-radius: 3px;
    opacity: 0.7;
}

.footer {
    background: linear-gradient(135deg, var(--quantum-red) 0%, var(--darker-red) 100%);
    color: var(--pure-white);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-top: 24px;
    font-weight: 500;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(216, 58, 58, 0.15);
}

/* Override label text for better readability */
label {
    color: var(--soft-black) !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
}

/* Slider styling */
input[type=range] {
    accent-color: var(--quantum-red) !important;
}

/* Radio buttons */
input[type=radio]:checked {
    background-color: var(--quantum-red) !important;
    border-color: var(--quantum-red) !important;
}

/* Checkbox styling */
input[type=checkbox]:checked {
    background-color: var(--quantum-red) !important;
    border-color: var(--quantum-red) !important;
}

/* Add subtle animations */
.container-box, .info-box {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.container-box:hover, .info-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
}
"""

with gr.Blocks(title="HED to Anything Webcam", css=custom_css) as demo:
    gr.HTML("""
    <div class="app-header">
        <h1>✨ Quantum Edge Detection Studio ✨</h1>
        <p>Transform your reality with elegant edge detection and neural style transfer</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="container-box">')
            gr.HTML("<h3>✦ Control Panel</h3>")
            
            gr.HTML('<div class="divider"></div>')
            
            blend_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.1,  # Set default to 0.1 (10% webcam opacity)
                step=0.1,
                label="Input Overlay Opacity (0 = natural edges only, 1 = full original image)"
            )
            
            cyclegan_toggle = gr.Checkbox(
                value=True,
                label="Enable CycleGAN Transformation"
            )
            
            cyclegan_direction = gr.Radio(
                choices=["Edges to Image", "Image to Edges"],
                value="Edges to Image",
                label="CycleGAN Direction"
            )
            
            desktop_mode = gr.Checkbox(
                value=False,
                label="Capture Desktop Instead of Webcam"
            )
            
            gr.HTML('<div class="divider"></div>')
            
            start_button = gr.Button("✧ Start Virtual Camera ✧", variant="primary")
            
            output_status = gr.Textbox(
                label="Status",
                value="Ready to start...",
                interactive=False
            )
            
            # Add environment variable info
            gr.HTML("""
            <div class="info-box">
                <h4>✦ Environment Setup</h4>
                <p>This application requires the following environment variables:</p>
                <ul>
                    <li><b>HED_PATH:</b> Path to the HED model directory (required)</li>
                    <li><b>CYCLEGAN_PATH:</b> Path to the CycleGAN directory (optional)</li>
                </ul>
                <p>Example: <code>export HED_PATH=/path/to/hed</code></p>
            </div>
            """)
            
            gr.HTML('</div>') # Close container-box
            
            gr.HTML("""
            <div class="info-box">
                <h4>✦ How It Works</h4>
                <p>This application merges two elegant technologies:</p>
                <ul>
                    <li><b>HED (Holistically-Nested Edge Detection):</b> Creates refined, detailed edge maps from your input</li>
                    <li><b>CycleGAN:</b> Transforms between domains with sophisticated style transfer techniques</li>
                </ul>
                <p>Experience a real-time artistic transformation of your reality through the lens of edge detection.</p>
            </div>
            """)
        
        with gr.Column():
            gr.HTML('<div class="container-box">')
            gr.HTML("<h3>✦ Instructions</h3>")
            
            gr.Markdown("""
            ### Getting Started:
            1. Adjust the blend strength - this controls how much of your original input is visible
               - 0.0 = natural edge detection only (grayscale edges)
               - 1.0 = original image fully visible
               - Default (0.1) shows mainly the edge map with just a hint of the original
            
            2. Configure CycleGAN:
               - Enable/disable CycleGAN transformation
               - **Edges to Image:** Converts edge maps to photorealistic images
               - **Image to Edges:** Creates stylized edge maps from your input
            
            3. Select Capture Mode:
               - **Webcam Mode:** Captures video from your webcam (default)
               - **Desktop Mode:** Captures your entire screen instead of webcam
            
            4. Click "Start Virtual Camera" to begin:
               - A preview window will appear showing the processed output
               - The same output is sent to a virtual camera for use in other applications
               - Press 'q' in the preview window to stop processing
            
            ### Requirements:
            - pyvirtualcam must be installed
            - OBS Virtual Camera (or similar) must be configured on your system
            - For Desktop capture: pyautogui or mss package must be installed
            - Environment variables must be properly set (see Environment Setup)
            
            ### Performance Tips:
            - The HED model requires significant processing power
            - Desktop capture may be slower than webcam capture
            - Reduce screen resolution for better performance in desktop mode
            """)
            
            gr.HTML('</div>') # Close container-box
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <p>✧ Quantum Edge Detection Studio | Elegantly Transform Your Reality ✧</p>
    </div>
    """)
    
    def start_processing(blend_alpha, enable_cyclegan, cyclegan_dir, use_desktop):
        try:
            result = process_webcam_with_hed_and_cyclegan(
                blend_alpha, 
                cyclegan_dir,
                enable_cyclegan,
                use_desktop
            )
            return result if result else "Processing completed. (If this message appears immediately, check for errors in the console)"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    start_button.click(
        fn=start_processing,
        inputs=[blend_slider, cyclegan_toggle, cyclegan_direction, desktop_mode],
        outputs=output_status
    )

if __name__ == "__main__":
    # Print environment variable information
    print(f"HED_PATH: {hed_path}")
    print(f"CYCLEGAN_PATH: {CYCLEGAN_PATH}")
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for processing.")
    else:
        print("WARNING: CUDA is not available. Processing will be very slow on CPU.")
    
    # Launch the app
    demo.launch()