import os.path as osp
import argparse
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# Argument parser
parser = argparse.ArgumentParser(description="Run ESRGAN model on an image.")
parser.add_argument("--input", type=str, required=True, help="Path to the input image or folder.")
parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
parser.add_argument("--clarity", type=float, default=1.0, help="Clarity multiplier (default: 1.0).")
parser.add_argument("--size", type=str, default="original", help="Output size: 'original', '2x', or '4x'.")
args = parser.parse_args()

# Paths and device configuration
model_path = '/Users/tushal/Desktop/new_project/ESRGAN/models/RRDB_ESRGAN_x4.pth'
device = torch.device("cpu")

# Load ESRGAN model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print(f"Model loaded from {model_path}.")
print("Processing...")

# Handle single file or batch processing
if osp.isfile(args.input):
    test_img_paths = [args.input]
else:
    test_img_paths = glob.glob(osp.join(args.input, "*"))

# Process each image
for idx, path in enumerate(test_img_paths, start=1):
    base, ext = osp.splitext(osp.basename(path))
    print(f"Processing {idx}: {base}{ext}")

    # Read and preprocess image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * args.clarity / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    # Resize if required
    if args.size == "2x":
        output = cv2.resize(output, (0, 0), fx=2, fy=2)
    elif args.size == "4x":
        output = cv2.resize(output, (0, 0), fx=4, fy=4)

    # Save output
    output_path = osp.join(args.output, f"{base}(1){ext}")
    cv2.imwrite(output_path, output)
    print(f"Saved output image: {output_path}")