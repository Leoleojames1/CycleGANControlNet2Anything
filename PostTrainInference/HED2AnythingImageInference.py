import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import sys

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

# Get CycleGAN path from environment variable or use default
CYCLEGAN_PATH = os.getenv('CYCLEGAN_PATH', '../TrainNotebooksCycleGAN/TrainWithHEDDataset')
print(f"Using CycleGAN path: {CYCLEGAN_PATH}")

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

# Custom CSS with quantum-inspired soft red aesthetic
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

/* Make the input and output image areas more prominent */
.image-display {
    border: 2px solid var(--quantum-red) !important;
    border-radius: 5px !important;
    padding: 2px !important;
    background-color: white !important;
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

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# Image postprocessing
def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy()
    return (tensor * 255).astype(np.uint8)

# Load HED model
def load_hed_model():
    print("Initializing HED model...")
    model = Network()
    if torch.cuda.is_available():
        model = model.cuda()
    model.train(False)
    print("HED model loaded!")
    return model

# Load generator model
def load_generator_model(model_path):
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
    model.eval()
    return model

# Process image with HED model
def process_with_hed(input_image, hed_model):
    try:
        # Convert to RGB if needed
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
            
        # Resize image (HED model expects specific dimensions)
        img_resized = input_image.resize((480, 320), Image.LANCZOS)
        
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
        output_img = Image.fromarray(output_np)
        
        # Resize back to original dimensions
        output_img = output_img.resize(input_image.size, Image.LANCZOS)
            
        return output_img
        
    except Exception as e:
        print(f"Error in HED processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Main transformation function
def transform_image(input_image, direction, generate_edges_only=False):
    if input_image is None:
        print("No input image provided")
        return None, "Please provide an input image"
        
    try:
        # Ensure input image is a PIL Image
        if isinstance(input_image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(input_image.shape) == 2:  # Grayscale
                input_image = np.stack([input_image] * 3, axis=-1)
            elif input_image.shape[-1] == 4:  # RGBA
                input_image = input_image[..., :3]
            input_image = Image.fromarray(input_image.astype('uint8'), 'RGB')
        
        # Initialize HED model
        hed_model = load_hed_model()
        
        # If user just wants edge detection, return HED output
        if generate_edges_only:
            edge_map = process_with_hed(input_image, hed_model)
            return edge_map, "Edge map generated successfully"
            
        # Get the correct model path based on direction
        if direction == "Edges to Image":
            model_path = os.path.join(CYCLEGAN_PATH, "checkpoints/hed2image/latest_net_G_A.pth")
        else:
            model_path = os.path.join(CYCLEGAN_PATH, "checkpoints/hed2image/latest_net_G_B.pth")
        
        # For "Image to Edges to Image" mode, first create edge map
        if direction == "Image to Edges to Image":
            # Generate edge map
            edge_map = process_with_hed(input_image, hed_model)
            
            # Now use the edge map as input to the "Edges to Image" model
            model_path = os.path.join(CYCLEGAN_PATH, "checkpoints/hed2image/latest_net_G_A.pth")
            input_image = edge_map
        
        # Load generator model
        generator = load_generator_model(model_path)
        if generator is None:
            return None, f"Failed to load model from {model_path}"
            
        # Preprocess image
        input_tensor = preprocess_image(input_image)
        
        # Generate output
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # Convert to image
        output_image = postprocess_image(output_tensor)
        output_pil = Image.fromarray(output_image)
        
        return output_pil, "Transformation completed successfully"
        
    except Exception as e:
        print(f"Error in transform_image: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# Create the Gradio interface
def create_demo():
    with gr.Blocks(title="HED to Anything Image Translation", css=custom_css) as demo:
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>✨ Quantum Edge Image Translation ✨</h1>
            <p>Transform between realistic images and edge maps with precision</p>
        </div>
        """)
        
        # Main interface
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                gr.HTML('<div class="container-box">')
                gr.HTML("<h3>✦ Control Panel</h3>")
                
                # Divider
                gr.HTML('<div class="divider"></div>')
                
                # Input controls
                with gr.Group():
                    gr.HTML('<b>IMAGE INPUT</b>')
                    input_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        elem_classes=["image-display"]
                    )
                
                # Divider
                gr.HTML('<div class="divider"></div>')
                
                # Mode selection
                with gr.Group():
                    gr.HTML('<b>TRANSFORMATION MODE</b>')
                    direction = gr.Radio(
                        choices=["Edges to Image", "Image to Edges", "Image to Edges to Image", "Generate Edges Only"],
                        value="Edges to Image",
                        label="Select Operation Mode"
                    )
                
                # Divider
                gr.HTML('<div class="divider"></div>')
                
                # Process button
                transform_btn = gr.Button("✧ Transform Image ✧", variant="primary")
                
                # Status output
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready to transform. Please upload an image and select mode."
                )
                
                # Environment variables info
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
                
                # Mode explanations
                gr.HTML("""
                <div class="info-box">
                    <h4>✦ Mode Explanations</h4>
                    <ul>
                        <li><b>Edges to Image:</b> Converts edge maps to photorealistic images</li>
                        <li><b>Image to Edges:</b> Extracts edge maps from photorealistic images</li>
                        <li><b>Image to Edges to Image:</b> Reconstructs a new image based on the edge structure of the input</li>
                        <li><b>Generate Edges Only:</b> Creates a high-quality edge map using HED without transformation</li>
                    </ul>
                </div>
                """)
                
                gr.HTML('</div>') # Close container-box
            
            # Right column - Output
            with gr.Column(scale=1):
                gr.HTML('<div class="container-box">')
                gr.HTML("<h3>✦ Result Preview</h3>")
                
                # Output image
                output_image = gr.Image(
                    label="Generated Output",
                    type="pil",
                    elem_classes=["image-display"]
                )
                
                # Technical explanation
                gr.HTML("""
                <div class="info-box">
                    <h4>✦ Technical Details</h4>
                    <p>This application uses a combination of:</p>
                    <ul>
                        <li><b>HED (Holistically-Nested Edge Detection):</b> A deep learning approach that produces high-quality edge maps by capturing both local and global information</li>
                        <li><b>CycleGAN Architecture:</b> An unpaired image-to-image translation network that learns to convert between domains without direct paired examples</li>
                        <li><b>ResNet Generator:</b> A deep residual network that maintains structural information while performing image translation</li>
                    </ul>
                    <p>For optimal results, use clear images with well-defined subjects against contrasting backgrounds.</p>
                </div>
                """)
                
                # Tips
                gr.HTML("""
                <div class="info-box">
                    <h4>✦ Tips for Best Results</h4>
                    <ul>
                        <li>When using "Edges to Image" mode, clean edge maps with clear structures produce the best results</li>
                        <li>For "Image to Edges" mode, photos with good lighting and clear subject separation work best</li>
                        <li>The "Image to Edges to Image" mode can create interesting artistic variations of the original image</li>
                        <li>Images are internally processed at 256x256 resolution for the generator</li>
                    </ul>
                </div>
                """)
                
                gr.HTML('</div>') # Close container-box
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p>✧ Quantum Edge Detection Studio | Elegantly Transform Your Reality ✧</p>
        </div>
        """)
        
        # Set up the processing function based on the selected direction
        def process_and_transform(image, mode):
            if mode == "Generate Edges Only":
                result_img, status = transform_image(image, mode, generate_edges_only=True)
            else:
                result_img, status = transform_image(image, mode)
                
            return result_img, status
            
        # Connect components
        transform_btn.click(
            fn=process_and_transform,
            inputs=[input_image, direction],
            outputs=[output_image, status_output]
        )
    
    return demo

if __name__ == "__main__":
    # Print environment variable information
    print(f"HED_PATH: {hed_path}")
    print(f"CYCLEGAN_PATH: {CYCLEGAN_PATH}")
    
    # Make sure checkpoints directory exists
    os.makedirs(os.path.join(CYCLEGAN_PATH, "checkpoints/hed2image"), exist_ok=True)
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for processing.")
    else:
        print("WARNING: CUDA is not available. Processing will be very slow on CPU.")
    
    # Create and launch the demo
    demo = create_demo()
    demo.queue(max_size=5).launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Set specific port
        show_error=True,        # Show detailed errors
        debug=True              # Enable debug mode
    )