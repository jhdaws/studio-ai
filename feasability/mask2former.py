import os
import torch
import torch_directml  # Import the DirectML library
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  # For visualizing the segmentation map

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)

# Set up DirectML device
device = torch_directml.device()  # Use DirectML device instead of CUDA
model.to(device)  # Move the model to the DirectML device
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((520, 520)),  # Resize to the input size expected by DeepLabV3
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

def segment_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)["out"][0]  # Get the output segmentation map
    
    # Convert the output to a segmentation map
    output_predictions = output.argmax(0).cpu().numpy()  # Move to CPU and convert to numpy array
    
    return output_predictions

if __name__ == "__main__":
    # Input image path
    image_path = input("Enter the path to the image: ")
    
    # Segment the image
    segmentation_map = segment_image(image_path)
    
    # Visualize the segmentation map
    plt.imshow(segmentation_map, cmap="nipy_spectral")  # Use a colormap to visualize classes
    plt.colorbar()
    plt.title("Segmentation Map")
    plt.show()
    
    # Optionally, save the segmentation map as an image
    segmentation_image = Image.fromarray((segmentation_map * 255 / segmentation_map.max()).astype(np.uint8))  # Scale to 0-255
    
    # Ensure the output directory exists
    output_dir = "image_out"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    output_path = os.path.join(output_dir, "segmentation_output.png")
    segmentation_image.save(output_path)
    print(f"Segmentation map saved as '{output_path}'")