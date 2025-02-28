from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import torch_directml  # Import the DirectML library

def describe_image(image_path):
    # Load the pre-trained BLIP-2 model and processor
    model_name = "Salesforce/blip2-opt-2.7b"  # You can also use "Salesforce/blip2-flan-t5-xl"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)

    # Set up DirectML device
    device = torch_directml.device()  # Use DirectML device instead of CUDA
    model.to(device)  # Move the model to the DirectML device

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image and generate text
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)  # Move inputs to DirectML device
    out = model.generate(**inputs)

    # Decode the generated text
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

if __name__ == "__main__":
    # Input image path
    image_path = input("Enter the path to the image: ")

    # Generate a description of the image
    description = describe_image(image_path)
    print("Image Description:", description)