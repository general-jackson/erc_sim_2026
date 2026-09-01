import glob
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==============================================================================
# SEGMENT 1: AUGMENTATION ENGINE
# Simulates variable camera angles, lighting conditions, blur, and contrast.
# ==============================================================================


def augment_digit_crop(gray_roi):
  """Applies random synthetic vision transformations to simulate Gazebo dynamic conditions."""

  # 1. Random contrast and brightness shift
  alpha = np.random.uniform(0.6, 1.4)  # Contrast gain multiplier
  beta = np.random.randint(-40, 40)  # Brightness offset
  augmented = cv2.convertScaleAbs(gray_roi, alpha=alpha, beta=beta)

  # 2. Binarize to white text on black background using Otsu thresholding
  _, binary = cv2.threshold(
      augmented, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )

  # 3. Random Affine Rotation (-12 degrees to +12 degrees)
  h, w = binary.shape
  angle = np.random.uniform(-12, 12)
  rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale=1.0)
  binary = cv2.warpAffine(
      binary,
      rotation_matrix,
      (w, h),
      borderMode=cv2.BORDER_CONSTANT,
      borderValue=0,
  )

  # 4. Resize and aspect-ratio padding into MNIST-standard 28x28 bounding box
  scale = 20.0 / max(h, w)
  nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
  resized = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_AREA)

  canvas = np.zeros((28, 28), dtype=np.uint8)
  dx, dy = (28 - nw) // 2, (28 - nh) // 2
  canvas[dy : dy + nh, dx : dx + nw] = resized

  # 5. Optional Gaussian Blur (simulate camera movement blur)
  if np.random.rand() > 0.5:
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

  return canvas


# ==============================================================================
# SEGMENT 2: PYTORCH DATASET LOADER
# Loads image paths, generates augmented batches, and converts images to Tensors.
# ==============================================================================


class GazeboDigitDataset(Dataset):

  def __init__(self, dataset_dir, samples_per_class=1000):
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    self.samples = []  # Holds image array tensors
    self.labels = []  # Holds corresponding target digit classes (1 to 5)

    # Loop through each target digit directory (1..5)
    for digit_class in range(1, 6):
      class_folder = os.path.join(dataset_dir, str(digit_class))
      image_paths = glob.glob(os.path.join(class_folder, "*.png")) + glob.glob(
          os.path.join(class_folder, "*.jpg")
      )

      if not image_paths:
        print(
          f"[Warning] No images found for digit class '{digit_class}' in"
          f" {class_folder}"
        )
        continue

      # Synthetically augment base images up to requested sample count per class
      for _ in range(samples_per_class):
        random_path = np.random.choice(image_paths)
        raw_crop = cv2.imread(random_path, cv2.IMREAD_GRAYSCALE)

        if raw_crop is None:
          continue

        processed_img = augment_digit_crop(raw_crop)

        # Normalize pixel values from range [0, 255] to floating point range [0.0, 1.0]
        normalized_img = processed_img.astype(np.float32) / 255.0

        # Add single channel dimension (Shape: 1 x 28 x 28)
        self.samples.append(normalized_img[np.newaxis, :, :])
        self.labels.append(digit_class)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    # Convert numpy arrays to PyTorch Tensors
    x_tensor = torch.tensor(self.samples[idx], dtype=torch.float32)
    y_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
    return x_tensor, y_tensor


# ==============================================================================
# SEGMENT 3: NEURAL NETWORK ARCHITECTURE
# A fast, lightweight 2-layer Convolutional Neural Network (CNN).
# ==============================================================================


class LightweightDigitCNN(nn.Module):

  def __init__(self):
    super().__init__()
    # Layer 1: Conv -> BatchNorm -> ReLU -> MaxPool (Input: 1x28x28 -> Output: 16x14x14)
    self.conv1 = nn.Conv2d(
        in_channels=1, out_channels=16, kernel_size=3, padding=1
    )
    self.bn1 = nn.BatchNorm2d(16)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Layer 2: Conv -> BatchNorm -> ReLU -> MaxPool (Input: 16x14x14 -> Output: 32x7x7)
    self.conv2 = nn.Conv2d(
        in_channels=16, out_channels=32, kernel_size=3, padding=1
    )
    self.bn2 = nn.BatchNorm2d(32)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Layer 3: Fully-Connected Classifier Head (Input: 32*7*7 -> Output: 6 classes [0..5])
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(32 * 7 * 7, 64)
    self.relu3 = nn.ReLU()
    self.dropout = nn.Dropout(0.25)
    self.fc2 = nn.Linear(64, 6)  # Output index matches target digit integer directly

  def forward(self, x):
    x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
    x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
    x = self.flatten(x)
    x = self.dropout(self.relu3(self.fc1(x)))
    x = self.fc2(x)
    return x


# ==============================================================================
# SEGMENT 4: TRAINING & ONNX EXPORT PIPELINE
# ==============================================================================

if __name__ == "__main__":
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset_raw")
  OUTPUT_ONNX_PATH = os.path.join(SCRIPT_DIR, "gazebo_digit_model.onnx")

  # 1. Instanciate Dataset and PyTorch DataLoader
  print("[1/4] Augmenting dataset and generating training batches...")
  train_dataset = GazeboDigitDataset(
      dataset_dir=DATASET_PATH, samples_per_class=1000
  )
  if len(train_dataset) == 0:
    raise RuntimeError(
        f'No training images found in {os.path.abspath(DATASET_PATH)}. '
        'Expected PNG or JPG files inside class folders 1 through 5.'
    )
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

  # 2. Instantiate Model, Loss Function, and Optimizer
  model = LightweightDigitCNN()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # 3. Model Training Loop
  print("[2/4] Starting CNN Training...")
  model.train()
  epochs = 12

  for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
      optimizer.zero_grad()  # Reset gradient accumulators

      outputs = model(images)  # Forward pass
      loss = criterion(outputs, labels)  # Compute loss
      loss.backward()  # Backpropagation
      optimizer.step()  # Update network weights

      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    avg_loss = running_loss / len(train_loader)
    print(
        f"Epoch [{epoch+1:02d}/{epochs:02d}] -> Loss: {avg_loss:.4f} | Accuracy:"
        f" {accuracy:.2f}%"
    )

  # 4. Export Model to ONNX Format
  print("[3/4] Exporting trained model to ONNX runtime format...")
  model.eval()

  # Create dummy input tensor matching input network dimensions (1 batch, 1 channel, 28x28)
  dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)

  torch.onnx.export(
      model,
      dummy_input,
      OUTPUT_ONNX_PATH,
      export_params=True,  # Save trained weights inside the ONNX model file
      opset_version=11,  # Standard OpenCV-compatible ONNX opset version
      do_constant_folding=True,  # Optimize model constants
      input_names=["input"],  # Input node identification name
      output_names=["output"],  # Output node identification name
        dynamo=False,  # Use the legacy exporter; the dynamo exporter requires onnxscript
  )

  print(
      f"[4/4] Model successfully trained and saved to: {OUTPUT_ONNX_PATH}!"
  )