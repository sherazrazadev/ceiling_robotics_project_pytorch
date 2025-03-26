import torch
from models_training import Policy
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def load_model(model_path, config):
    model = Policy(config, rank=0, world_size=1)  # Use rank=0 and world_size=1 for single GPU or CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to appropriate dimensions if necessary
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def main(args):
    # Define configuration based on the training configuration
    config_defaults = {
        "feedback_type": args.feedback_type,
        "task": args.task,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 224,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 2
    }

    # Load model
    model = load_model(args.model_path, config_defaults)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Preprocess image
    image = preprocess_image(args.image_path)

    # Move image to the same device as model
    image = image.to(device)

    # Prepare dummy proprioception and LSTM state (replace with real data if available)
    dummy_proprio = torch.zeros(1, config_defaults["proprio_dim"], device=device)  # (batch_size, proprio_dim)
    lstm_state = None  # Initial state for LSTM

    # Perform inference
    with torch.no_grad():
        action, _ = model.predict(image.squeeze(0), dummy_proprio, lstm_state)
    
    # Output the action prediction
    print("Predicted Action:", action)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="5dstator",
        help="Task used during training"
    )
    parser.add_argument(
        "--feedback_type",
        type=str,
        default="cloning_100",
        help="Feedback type used during training"
    )
    args = parser.parse_args()
    main(args)


#for best.pt integration
import torch
from models_training import Policy
from argparse import ArgumentParser
from PIL import Image
import torchvision.transforms as transforms
import cv2
from ultralytics import YOLO

def load_model(model_path, config):
    model = Policy(config, rank=0, world_size=1)  # Use rank=0 and world_size=1 for single GPU or CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to appropriate dimensions if necessary
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def detect_and_draw_bounding_box(image_path, model, save_image=False):
    # Load image
    image = cv2.imread(image_path)
    
    # Perform object detection
    results = model(image_path)

    # Extract bounding boxes and confidences
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    if len(confidences) == 0:
        print("No objects detected.")
        return None

    # Find the box with the highest confidence
    max_confidence_index = confidences.argmax()
    max_confidence_box = boxes[max_confidence_index]

    # Draw the bounding box on the image
    x1, y1, x2, y2 = map(int, max_confidence_box[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Optionally save the image
    if save_image:
        cv2.imwrite("output_with_bbox.jpg", image)
    
    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    return image_pil

def main(args):
    # Define configuration based on the training configuration
    config_defaults = {
        "feedback_type": args.feedback_type,
        "task": args.task,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 224,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 2
    }

    # Load model
    model = load_model(args.model_path, config_defaults)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    yolo_model = YOLO("best.pt")  # Replace with your specific YOLO model path

    # Detect object and draw bounding box
    image_with_bbox = detect_and_draw_bounding_box(args.image_path, yolo_model, save_image=args.save_image)
    
    if image_with_bbox is None:
        print("No objects detected, skipping action prediction.")
        return

    print("Image with bounding box processed.")

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to appropriate dimensions if necessary
        transforms.ToTensor(),
    ])
    image = transform(image_with_bbox)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move image to the same device as model
    image = image.to(device)

    # Prepare dummy proprioception and LSTM state
    dummy_proprio = torch.zeros(1, config_defaults["proprio_dim"], device=device)  # (batch_size, proprio_dim)
    lstm_state = None  # Initial state for LSTM

    # Perform inference
    with torch.no_grad():
        action, _ = model.predict(image.squeeze(0), dummy_proprio, lstm_state)
    
    # Output the action prediction
    print("Predicted Action:", action)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="5dstator",
        help="Task used during training"
    )
    parser.add_argument(
        "--feedback_type",
        type=str,
        default="cloning_100",
        help="Feedback type used during training"
    )
    parser.add_argument(
        "--save_image",
        action='store_true',
        help="Save the image with bounding box"
    )
    args = parser.parse_args()
    main(args)
