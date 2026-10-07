import torch
import config
import os
import utils
from tqdm import tqdm
from VOC import VOCDataset, voc_collate_fn, idx_to_class
from YOLO import *
from torch.utils.data import DataLoader


MODEL_DIR = 'models/yolo_v1/04_22_2026/22_41_36' # CHANGE THIS TO YOUR MODEL DIRECTORY

def validate():
    dataset = VOCDataset(root='data', train=False)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        collate_fn=voc_collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv1ResNet()  
    model.eval()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'weights', 'best')))

    model = model.to(device)  

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for image, original, _ in tqdm(loader):
            image = image.to(device)
            predictions = model.forward(image)
            all_predictions.append(predictions)
            all_targets.append(original)
    
    # concatenate everything
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # compute metrics ONCE
    metrics = utils.evaluate(all_predictions, all_targets)
    print(metrics)




if __name__ == '__main__':
    validate()
