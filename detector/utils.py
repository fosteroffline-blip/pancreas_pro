import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
pancreas_model = torch.load('pancreas_model.pth', map_location=device)
tumor_model = torch.load('tumor_model.pth', map_location=device)

pancreas_model.eval()
tumor_model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# Stage 1: Pancreas Detection
# -------------------------
def detect_pancreas(image_path):
    image = Image.open(image_path).convert('RGB')
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = pancreas_model(img)
        pred = torch.argmax(output, dim=1).item()

    return pred


# -------------------------
# Stage 2: Cancer Detection
# -------------------------
def detect_cancer(image_path):
    image = Image.open(image_path).convert('RGB')
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = tumor_model(img)
        pred = torch.argmax(output, dim=1).item()

    return pred


# -------------------------
# Grad-CAM
# -------------------------
def generate_gradcam(model, image_path, save_path):
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    input_img = transform(Image.fromarray(img)).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # IMPORTANT: change based on your model
    target_layer = None
    for name, module in model.named_modules():
        if "conv" in name:
            target_layer = module

    if target_layer is None:
        raise Exception("No conv layer found!")

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(input_img)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img

    cv2.imwrite(save_path, overlay)

    return save_path