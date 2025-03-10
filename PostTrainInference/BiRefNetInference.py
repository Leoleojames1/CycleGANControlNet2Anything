import gradio as gr
import torch
from transformers import AutoModelForImageSegmentation
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from torchvision import transforms
import argparse

# Load the model
model = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create custom feature extractor since the model doesn't have a standard preprocessor_config.json
class CustomFeatureExtractor:
    def __init__(self):
        # Standard image preprocessing for segmentation models
        self.resize_transform = transforms.Resize((512, 512), antialias=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, images, return_tensors="pt"):
        if isinstance(images, Image.Image):
            # Store original size
            orig_size = images.size
            
            # Convert to tensor keeping aspect ratio
            img_tensor = transforms.ToTensor()(images)
            
            # Create a square tensor with padding
            c, h, w = img_tensor.shape
            max_dim = max(h, w)
            pad_h = (max_dim - h) // 2
            pad_w = (max_dim - w) // 2
            
            padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
            padded_tensor = transforms.Pad(padding)(img_tensor)
            
            # Resize and normalize
            model_input = self.resize_transform(padded_tensor)
            model_input = self.normalize(model_input)
            model_input = model_input.unsqueeze(0)
            
            # Save original dimensions and padding for later
            padding_info = {
                'orig_size': orig_size, 
                'padded_size': (max_dim, max_dim),
                'padding': padding
            }
            
            return {"image": model_input, "padding_info": padding_info}
            
        else:
            # Handle case where input is not a PIL image
            raise ValueError("Input must be a PIL Image")

# Create custom feature extractor
feature_extractor = CustomFeatureExtractor()

# Constants for visualization
PALETTE = np.array([
    [0, 0, 0],         # Background
    [128, 0, 0],       # Class 1
    [0, 128, 0],       # Class 2
    [128, 128, 0],     # Class 3
    [0, 0, 128],       # Class 4
    [128, 0, 128],     # Class 5
    [0, 128, 128],     # Class 6
    [128, 128, 128],   # Class 7
    [64, 0, 0],        # Class 8
    [192, 0, 0],       # Class 9
    [64, 128, 0],      # Class 10
    [192, 128, 0],     # Class 11
    [64, 0, 128],      # Class 12
    [192, 0, 128],     # Class 13
    [64, 128, 128],    # Class 14
    [192, 128, 128],   # Class 15
    [0, 64, 0],        # Class 16
    [128, 64, 0],      # Class 17
    [0, 192, 0],       # Class 18
    [128, 192, 0],     # Class 19
])

def get_segmentation_mask(image):
    """Process input image and return the raw segmentation mask"""
    # Prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    image_tensor = inputs["image"].to(device)
    padding_info = inputs["padding_info"]
    
    # Inference - pass the tensor directly instead of as a keyword argument
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Debug output
    print(f"Model output type: {type(outputs)}")
    if isinstance(outputs, list):
        print(f"List length: {len(outputs)}")
        for i, item in enumerate(outputs):
            print(f"Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
            if i == 0:  # Print min/max values
                print(f"Item {i} min: {item.min().item()}, max: {item.max().item()}")
    
    # Get the mask from model output (single channel)
    if isinstance(outputs, list):
        mask_tensor = outputs[0]  # Get the first item from the list
    else:
        mask_tensor = outputs
    
    # Convert to binary mask (0 and 1)
    if mask_tensor.shape[1] == 1:  # Single channel
        mask = mask_tensor.squeeze(1).cpu().numpy()[0]
        binary_mask = (mask > 0.5).astype(np.uint8)
    else:  # Multiple channels
        binary_mask = mask_tensor.argmax(dim=1).cpu().numpy()[0]
    
    # Resize back to padded size before removing padding
    mask_padded = transforms.ToPILImage()(torch.tensor(binary_mask).unsqueeze(0).float())
    mask_padded = mask_padded.resize((padding_info['padded_size'][1], padding_info['padded_size'][0]), Image.NEAREST)
    
    # Remove padding to get original aspect ratio
    orig_w, orig_h = padding_info['orig_size']
    padding = padding_info['padding']
    
    # Calculate the crop box (left, upper, right, lower)
    left = padding[0]
    upper = padding[1]
    right = padding_info['padded_size'][1] - padding[2]
    lower = padding_info['padded_size'][0] - padding[3]
    
    # Crop the mask
    mask_orig_ratio = mask_padded.crop((left, upper, right, lower))
    
    # Resize to original dimensions
    mask_orig_size = mask_orig_ratio.resize((orig_w, orig_h), Image.NEAREST)
    
    # Convert back to numpy for processing
    mask_final = np.array(mask_orig_size)
    
    return mask_final

def create_transparent_background(image, mask):
    """
    Create a version of the image with transparent background.
    Args:
        image: PIL Image
        mask: Segmentation mask (numpy array)
    Returns:
        PIL Image with transparent background
    """
    # Convert to numpy array
    np_image = np.array(image)
    
    # Create an RGBA image
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:  # RGB
        rgba_image = np.zeros((np_image.shape[0], np_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = np_image
        
        # Make sure mask is binary
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Set alpha channel based on mask (255 for foreground, 0 for background)
        rgba_image[:, :, 3] = binary_mask * 255
    else:  # Already has alpha or is grayscale
        if len(np_image.shape) == 2:  # Grayscale
            np_image = np.stack([np_image] * 3, axis=-1)
            
        # Create RGBA
        rgba_image = np.zeros((np_image.shape[0], np_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = np_image[:, :, :3]
        
        # Make sure mask is binary
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Set alpha channel
        rgba_image[:, :, 3] = binary_mask * 255
    
    # Print debug info
    print(f"Alpha channel min: {rgba_image[:,:,3].min()}, max: {rgba_image[:,:,3].max()}")
    print(f"Mask min: {mask.min()}, max: {mask.max()}, sum: {np.sum(mask)}")
    
    return Image.fromarray(rgba_image)

def create_visualization(image, mask, transparent=None, blended=None, mask_color=(255, 0, 0)):
    """
    Create a visualization image showing the process.
    All images must be at the original aspect ratio/size.
    
    Args:
        image: Original PIL image
        mask: Binary segmentation mask
        transparent: PIL image with transparent background
        blended: PIL image with blended overlay
        mask_color: RGB tuple for mask color (default: red)
    """
    # Get original dimensions
    width, height = image.size
    
    # Determine max width to prevent output from being too large
    max_width = 1200
    if width * 4 > max_width:  # 4 images side by side
        scale_factor = max_width / (width * 4)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    else:
        new_width = width
        new_height = height
        
    # Convert mask to RGB visualization for display with the specified color
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask > 0] = mask_color  # Use the specified mask color
    mask_rgb_img = Image.fromarray(mask_rgb)
    
    # Resize everything to consistent size for display
    image_resized = image.resize((new_width, new_height), Image.LANCZOS)
    mask_resized = mask_rgb_img.resize((new_width, new_height), Image.NEAREST)
    
    if transparent:
        transparent_resized = transparent.resize((new_width, new_height), Image.LANCZOS)
    else:
        transparent_resized = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
        
    if blended:
        blended_resized = blended.resize((new_width, new_height), Image.LANCZOS)
    else:
        blended_resized = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
    
    # Create a combined visualization image
    result_width = new_width * 4
    result_height = new_height
    result = Image.new('RGBA', (result_width, result_height), (255, 255, 255, 255))
    
    # Paste images
    result.paste(image_resized, (0, 0))
    result.paste(mask_resized, (new_width, 0))
    result.paste(transparent_resized, (new_width * 2, 0))
    result.paste(blended_resized, (new_width * 3, 0))
    
    # Add labels
    return result

def create_blended_view(image, mask, mask_color=(255, 0, 0)):
    """
    Create a blended view of the mask overlaid on the original image
    
    Args:
        image: PIL Image of the original
        mask: Binary segmentation mask
        mask_color: RGB tuple for mask color
    """
    # Convert PIL to numpy arrays
    img_np = np.array(image).astype(np.float32) / 255.0
    
    # Create a colored mask for overlay with the specified color
    overlay = np.zeros_like(img_np)
    
    # Convert RGB to normalized values
    r, g, b = mask_color
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    
    # Apply the color to the overlay
    overlay[mask > 0, 0] = r_norm  # Red channel
    overlay[mask > 0, 1] = g_norm  # Green channel 
    overlay[mask > 0, 2] = b_norm  # Blue channel
    
    # Blend the images (70% original, 30% overlay)
    alpha = 0.7
    blended = (alpha * img_np + (1-alpha) * overlay)
    blended = (blended * 255).astype(np.uint8)
    
    return Image.fromarray(blended)

def process_image(image, output_mode, mask_color=(255, 0, 0)):
    """
    Process image based on selected output mode
    
    Args:
        image: Input PIL image
        output_mode: Processing mode (Standard/Remove Background/Blended)
        mask_color: RGB tuple for mask color
    """
    try:
        # Get the segmentation mask (now at original aspect ratio)
        mask = get_segmentation_mask(image)
        
        # Outputs for different modes
        if output_mode == "Standard Segmentation":
            return create_visualization(image, mask, mask_color=mask_color)
            
        elif output_mode == "Remove Background":
            # Create image with transparent background
            transparent = create_transparent_background(image, mask)
            return create_visualization(image, mask, transparent, mask_color=mask_color), transparent
            
        else:  # Blended View
            # Create blended view with overlay
            transparent = create_transparent_background(image, mask)
            blended = create_blended_view(image, mask, mask_color=mask_color)
            return create_visualization(image, mask, transparent, blended, mask_color=mask_color), None
            
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict(image, output_mode, mask_color_choice):
    """Gradio prediction function"""
    if image is None:
        return None
    
    # Convert to PIL image if it's not already
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert('RGB')
    
    # Convert color choice to RGB
    color_map = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "White": (255, 255, 255),
        "Black": (0, 0, 0)
    }
    
    mask_color = color_map.get(mask_color_choice, (255, 0, 0))  # Default to red
    
    if output_mode == "Standard Segmentation":
        return process_image(image, output_mode, mask_color)
    elif output_mode == "Remove Background":
        viz, transparent = process_image(image, output_mode, mask_color)
        return viz, transparent
    else:  # Blended View
        return process_image(image, output_mode, mask_color)[0], None

# Create Gradio interface
with gr.Blocks(title="BiRefNet Image Segmentation") as demo:
    gr.Markdown("# BiRefNet Image Segmentation")
    gr.Markdown("Upload an image to get its segmentation mask, remove the background, or see a blended view.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            
            with gr.Row():
                with gr.Column():
                    output_mode = gr.Radio(
                        ["Standard Segmentation", "Remove Background", "Blended View"],
                        label="Output Mode",
                        value="Standard Segmentation"
                    )
                
                with gr.Column():
                    mask_color_choice = gr.Dropdown(
                        ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White", "Black"],
                        label="Mask Color",
                        value="Red"
                    )
            
            submit_btn = gr.Button("Process Image")
        
        with gr.Column():
            output_visualization = gr.Image(type="pil", label="Visualization")
            output_transparent = gr.Image(type="pil", label="Transparent Image", visible=False)
    
    def update_visibility(mode):
        if mode == "Remove Background":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
    
    output_mode.change(fn=update_visibility, inputs=output_mode, outputs=output_transparent)
    
    submit_btn.click(
        fn=predict,
        inputs=[input_image, output_mode, mask_color_choice],
        outputs=[output_visualization, output_transparent]
    )

# Launch the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BiRefNet Image Segmentation App')
    parser.add_argument('--share', action='store_true', help='Create a public link for the app')
    args = parser.parse_args()
    
    print(f"Launching app with share={args.share}")
    demo.launch(share=args.share)