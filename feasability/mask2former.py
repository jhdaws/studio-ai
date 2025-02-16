import os
import torch
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# Load the pre-trained model and feature extractor
model_name = "facebook/maskformer-swin-base-coco"
feature_extractor = MaskFormerFeatureExtractor.from_pretrained(model_name)
model = MaskFormerForInstanceSegmentation.from_pretrained(model_name)

def segment_image(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the output
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    # Extract the segmentation map and labels
    segmentation = result["segmentation"]
    segments_info = result["segments_info"]
    
    # Print the detected objects
    for segment in segments_info:
        label_id = segment["label_id"]
        label = model.config.id2label[label_id]
        print(f"Detected {label} with segment ID {segment['id']}")
    
    return segmentation, segments_info

if __name__ == "__main__":
    # Input image path
    image_path = input("Enter the path to the image: ")
    
    # Segment the image
    segmentation, segments_info = segment_image(image_path)
    
    # Optionally, save the segmentation map
    segmentation_image = Image.fromarray(segmentation.numpy())
    
    # Ensure the output directory exists
    output_dir = "image_out"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    output_path = os.path.join(output_dir, "segmentation_output.png")
    segmentation_image.save(output_path)
    print(f"Segmentation map saved as '{output_path}'")