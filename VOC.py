import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import copy

# Constants
IMG_SIZE = (448, 448)
CENTER_IMAGE = (224, 224) 

# v2.ToTensor is deprecated; ToImage + ToDtype(scale=True) is the modern way
transform_pipeline = v2.Compose([
    v2.ToImage(),
    v2.Resize(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
])

# 1. Define Class Mapping
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class_to_idx = {name: i for i, name in enumerate(VOC_CLASSES)}
idx_to_class = {i: name for i, name in enumerate(VOC_CLASSES)}

def get_scaled_eclipse(obj, original_size):
    """Converts VOC dict bbox to scaled center-based eclipse coordinates."""
    # original_size is (height, width)
    orig_h, orig_w = original_size
    bbox = obj['bndbox']
    
    # Scale coordinates to 448x448
    xmin = float(bbox['xmin']) * (IMG_SIZE[0] / orig_w)
    ymin = float(bbox['ymin']) * (IMG_SIZE[1] / orig_h)
    xmax = float(bbox['xmax']) * (IMG_SIZE[0] / orig_w)
    ymax = float(bbox['ymax']) * (IMG_SIZE[1] / orig_h)
    
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    
    return [center_x, center_y, width, height, 0.0]

def decode_label_tensor(label_matrix, S=7, C=20, threshold=0.5):
    """
    Converts a YOLO label tensor back into a list of dictionaries.
    
    Args:
        label_matrix (torch.Tensor): Tensor of shape (S, S, C + B*6)
        S (int): Grid size
        C (int): Number of classes
        threshold (float): Confidence threshold to treat a cell as having an object
        
    Returns:
        List[dict]: List containing {'name': str, 'eclipse': [cx, cy, w, h, angle_deg]}
    """
    decoded_objects = []
    
    # Iterate through each grid cell
    for i in range(S):       # y-axis (row)
        for j in range(S):   # x-axis (col)
            
            # Confidence is at index C
            conf = label_matrix[i, j, C]
            
            if conf > threshold:
                # 1. Get Class Name
                class_probs = label_matrix[i, j, :C]
                class_idx = torch.argmax(class_probs).item()
                obj_name = idx_to_class[class_idx]
                
                # 2. Extract geometry [x_cell, y_cell, w_norm, h_norm, sin_a]
                # These are located at C+1 to C+5
                geom = label_matrix[i, j, C+1:C+6]
                x_cell, y_cell, w_norm, h_norm, sin_a = geom.tolist()
                
                # 3. Reverse the scaling logic
                # cx = (cell_index + offset_within_cell) * (total_pixels / S)
                cx = (j + x_cell) * (IMG_SIZE[0] / S)
                cy = (i + y_cell) * (IMG_SIZE[1] / S)
                
                # w, h = norm_dim * total_pixels
                w = w_norm * IMG_SIZE[0]
                h = h_norm * IMG_SIZE[1]
                
                # 4. Reverse the angle (Sine to Degrees)
                # Clip sin_a to [-1, 1] to avoid NaNs from floating point jitter
                angle_rad = np.arcsin(np.clip(sin_a, -1.0, 1.0))
                angle_deg = np.rad2deg(angle_rad)
                
                decoded_objects.append({
                    'name': obj_name,
                    'eclipse': [cx, cy, w, h, angle_deg]
                })
                
    return decoded_objects


class VOCDataset(Dataset):
    def __init__(self, root, year='2012', train=True, num_samples=None, transform=transform_pipeline):
        image_set = 'train' if train else 'val'
        self.dataset = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform
        self.num_samples = num_samples

    def __len__(self):
        if(self.num_samples is not None):
            return min(len(self.dataset) * 3, self.num_samples * 3)  # Each sample generates 3 variants
        return len(self.dataset) * 3

    def __getitem__(self, idx):
        actual_idx = idx // 3
        variant = idx % 3  # 0: Original, 1: Left, 2: Right

        if actual_idx >= len(self.dataset):
            raise IndexError("Index out of range for the dataset.")
        
        # 1. Deepcopy the target to prevent cross-contamination between indices
        image, raw_target = self.dataset[actual_idx]
        target = copy.deepcopy(raw_target)
        orig_size = (image.height, image.width)

        # 2. Apply Image Scaling
        if self.transform:
            image = self.transform(image)

        # 3. Scale all coordinates to 448x448 first
        for obj in target['annotation']['object']:
            obj['eclipse'] = get_scaled_eclipse(obj, orig_size)

        # 4. Handle Rotations
        if variant > 0:
            angle = np.random.uniform(0, 20) if variant == 1 else np.random.uniform(-20, 0)
            image, target = self._apply_rotation(image, target, angle)

        label_tensor = self.encode(target)
        return image, label_tensor, target

    def _apply_rotation(self, image, target, angle):
        # Rotate Image once
        RM = cv2.getRotationMatrix2D(CENTER_IMAGE, angle, 1.0)
        img_np = image.permute(1, 2, 0).numpy()
        rotated_img = cv2.warpAffine(img_np, RM, IMG_SIZE)
        image = torch.from_numpy(rotated_img).permute(2, 0, 1)

        valid_objects = []
        for obj in target['annotation']['object']:
            cx, cy, w, h, _ = obj['eclipse']
            
            # Rotate point
            point = np.array([[[cx, cy]]], dtype=np.float32)
            new_p = cv2.transform(point, RM)[0][0]
            
            # Clip center to image bounds
            new_cx = np.clip(new_p[0], 0, IMG_SIZE[0])
            new_cy = np.clip(new_p[1], 0, IMG_SIZE[1])
            
            # 3. VISIBILITY CHECK: Only keep if center is inside the frame
            # We check the 0 to 448 (IMG_SIZE) boundaries
            is_inside = (0 <= new_cx <= IMG_SIZE[0]) and (0 <= new_cy <= IMG_SIZE[1])

            if is_inside:
                # Note: We don't even need np.clip here now, 
                # as the check ensures it's within bounds.
                obj['eclipse'] = (new_cx, new_cy, w, h, -angle) 
                valid_objects.append(obj)
            
        target['annotation']['object'] = valid_objects
        return image, target

    # def encode(self, target, S=7, B=2, C=20):
    #     """
    #     Encodes target into a tensor of shape (S, S, C + B*6)
    #     Each box is [center_x, center_y, width, height, confidence]
    #     """
    #     label_matrix = torch.zeros((S, S, C + B * 6))
        
    #     for obj in target['annotation']['object']:
    #         class_idx = class_to_idx[obj['name']]
    #         cx, cy, w, h, angle_deg = obj['eclipse']
            
    #         i, j = int(S * cy / IMG_SIZE[1]), int(S * cx / IMG_SIZE[0])
            
    #         if label_matrix[i, j, C] == 0:
    #             label_matrix[i, j, C] = 1 # Conf
    #             x_cell = (S * cx / IMG_SIZE[0]) - j
    #             y_cell = (S * cy / IMG_SIZE[1]) - i
                
    #             # Convert angle to sin(radians) for the network
    #             sin_a = np.sin(np.deg2rad(angle_deg))
                
    #             label_matrix[i, j, C+1:C+6] = torch.tensor([x_cell, y_cell, w/448, h/448, sin_a])
    #             label_matrix[i, j, class_idx] = 1
                
    #     return label_matrix

    def encode(self, target, S=7, B=2, C=20):
        label_matrix = torch.zeros((S, S, C + B * 6))
        
        for obj in target['annotation']['object']:
            class_idx = class_to_idx[obj['name']]
            cx, cy, w, h, angle_deg = obj['eclipse']
            
            i = min(int(S * cy / IMG_SIZE[1]), S - 1)  # clamp to [0, S-1]
            j = min(int(S * cx / IMG_SIZE[0]), S - 1)  # clamp to [0, S-1]
            
            if label_matrix[i, j, C] == 0:
                label_matrix[i, j, C] = 1
                x_cell = (S * cx / IMG_SIZE[0]) - j
                y_cell = (S * cy / IMG_SIZE[1]) - i
                
                sin_a = np.sin(np.deg2rad(angle_deg))
                
                label_matrix[i, j, C+1:C+6] = torch.tensor([x_cell, y_cell, w/448, h/448, sin_a])

                # Refill the second box slot with the same data for simplicity, to avoid the zeros problem
                label_matrix[i, j, C + 6] = 1
                label_matrix[i, j, C+7:C+12] = torch.tensor([x_cell, y_cell, w/448, h/448, sin_a])
                label_matrix[i, j, class_idx] = 1
                
        return label_matrix
    
def voc_collate_fn(batch):
    """
    Custom collate function for the VOCDataset.
    Stacks the fixed-size images and label tensors, but keeps the 
    variable-length target dictionaries in a list.
    """
    # Unzip the batch into separate lists
    images, label_tensors, targets = zip(*batch)
    
    # Stack the images into a single tensor of shape (N, C, H, W)
    images = torch.stack(images, dim=0)
    
    # Stack the label tensors into a single tensor of shape (N, S, S, C + B*6)
    label_tensors = torch.stack(label_tensors, dim=0)
    
    # Keep the raw target dictionaries as a tuple
    return images, label_tensors, targets

# # --- Verification and Plotting ---
# dataset = VOCDataset(root='data', train=False, transform=transform_pipeline)

# for i in range(9): # View first 3 images and their 3 variations
#     img, label_tensor, target = dataset[i]
#     fig, ax = plt.subplots(figsize=(5,5))
#     ax.imshow(img.permute(1, 2, 0))
#     print(f"Object Structure: {target['annotation']['object'][0]}")
#     print(f"Decoded Objects: {decode_label_tensor(label_tensor)}\n\n")
#     for obj in target['annotation']['object']:
#         ec = obj['eclipse']
#         # xy is the CENTER of the ellipse
#         ellipse = patches.Ellipse(
#             xy=(ec[0], ec[1]), width=ec[2], height=ec[3], angle=ec[4],
#             edgecolor='r', facecolor='none', linewidth=2
#         )
#         ax.add_patch(ellipse)
    
#     plt.title(f"Idx: {i} - {target['annotation']['object'][0]['name']}")
#     plt.axis('off')
#     plt.savefig(f'output_image_{i}.png')  # Save the image for verification