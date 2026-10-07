import torch
import config
import os
from utils import plot_detection_summary
from tqdm import tqdm
from VOC import VOCDataset, voc_collate_fn, idx_to_class
from YOLO import *
from torch.utils.data import DataLoader


MODEL_DIR = 'models/yolo_v1/04_22_2026/22_41_36'

def plot_test_images():
    classes = idx_to_class

    dataset = VOCDataset(root='data', train=False, num_samples=100)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        shuffle=True,
        drop_last=True,
        collate_fn=voc_collate_fn
    )

    model = YOLOv1ResNet()
    model.eval()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'weights', 'best')))

    count = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for image, original, _ in loader:
            predictions = model.forward(image)
            plot_detection_summary(image, original, predictions)
            break
    



if __name__ == '__main__':
    plot_test_images()
