import torch
import os
import numpy as np
import config
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from VOC import VOCDataset, voc_collate_fn
from LOSS import SumSquaredErrorLoss
from YOLO import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def hyperparameters_search(
        learning_rates=[5e-5, 1e-5, 5e-4, 1e-4],
        l_coords=[1, 1.5, 2, 2.5, 3],
        l_noobjs=[0.05, 0.1, 0.15, 0.2]):
    best_hyperparameters = {'learning_rate': None, 'l_coord': None, 'l_noobj': None}
    best_loss = float('inf')
    num_samples = 1000
    min_epochs = 5

    search_train_dataset = VOCDataset(
        root='data',
        train=True,
        num_samples=num_samples,
    )
    search_val_dataset = VOCDataset(
        root='data',
        train=False,
        num_samples=num_samples,
    )

    train_loader = DataLoader(
        search_train_dataset,
        batch_size=config.BATCH_SIZE * 2,  # Use a larger batch size for faster training during hyperparameter search
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        shuffle=True,
        collate_fn=voc_collate_fn
    )
    val_loader = DataLoader(
        search_val_dataset,
        batch_size=config.BATCH_SIZE * 2,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        collate_fn=voc_collate_fn
    )

    for lr in learning_rates:
        for l_coord in l_coords:
            for l_noobj in l_noobjs:
                print(f"Testing hyperparameters: lr={lr}, l_coord={l_coord}, l_noobj={l_noobj}")
                model = YOLOv1ResNet().to(device)
                loss_function = SumSquaredErrorLoss(l_coord=l_coord, l_noobj=l_noobj)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr
                )

                # Train for a few epochs and evaluate on the validation set
                for epoch in tqdm(range(min_epochs), desc='Epochs'):
                    model.train()
                    for data, labels, _ in tqdm(train_loader, desc='Train', leave=False):
                        data = data.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()
                        predictions = model.forward(data)
                        loss = loss_function(predictions, labels)
                        loss.backward()
                        optimizer.step()

                        del data, labels

                # Evaluate on the validation set
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data, labels, _ in tqdm(val_loader, desc='Val', leave=False):
                        data = data.to(device)
                        labels = labels.to(device)

                        predictions = model.forward(data)
                        loss = loss_function(predictions, labels)

                        val_loss += loss.item() / len(val_loader)
                        del data, labels

                print(f"Validation Loss: {val_loss}")
                if best_hyperparameters['learning_rate'] is None or val_loss < best_loss:
                    best_hyperparameters['learning_rate'] = lr
                    best_hyperparameters['l_coord'] = l_coord
                    best_hyperparameters['l_noobj'] = l_noobj
                    best_loss = val_loss
                del model, loss_function, optimizer
    
    print(f"Best Hyperparameters: {best_hyperparameters}, Loss: {best_loss}")
    return best_hyperparameters
                


    

if __name__ == '__main__':      # Prevent recursive subprocess creation
    torch.autograd.set_detect_anomaly(True)         # Check for nan loss
    # params = hyperparameters_search()
    # lr = params['learning_rate']
    # l_coord = params['l_coord']
    # l_noobj = params['l_noobj']

    lr = 0.0001
    l_coord = 1
    l_noobj = 0.05

    writer = SummaryWriter()
    now = datetime.now()

    model = YOLOv1ResNet().to(device)
    loss_function = SumSquaredErrorLoss(l_coord=l_coord, l_noobj=l_noobj)

    # Adam works better
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=0.9,
    #     weight_decay=5E-4
    # )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )


    # Load the dataset
    train_set = VOCDataset(root='data', train=True)
    test_set = VOCDataset(root='data', train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        shuffle=True,
        collate_fn=voc_collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        collate_fn=voc_collate_fn
    )

    # Create folders
    root = os.path.join(
        'models',
        'yolo_v1',
        now.strftime('%m_%d_%Y'),
        now.strftime('%H_%M_%S')
    )
    weight_dir = os.path.join(root, 'weights')
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)

    # Metrics
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    test_errors = np.empty((2, 0))


    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'test_losses'), test_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'test_errors'), test_errors)


    #####################
    #       Train       #
    #####################
    best_test_loss = float('inf')
    for epoch in tqdm(range(config.EPOCHS), desc='Epoch'):
        if epoch == config.WARMUP_EPOCHS:
            # unfreeze the last 2 layers of backbone after warmup
            for param in model.model[0].layer3.parameters():
                param.requires_grad = True
            for param in model.model[0].layer4.parameters():
                param.requires_grad = True

                
        model.train()
        train_loss = 0
        for data, labels, _ in tqdm(train_loader, desc='Train', leave=False):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            del data, labels

        # Step and graph scheduler once an epoch
        # writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        # scheduler.step()

        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
        writer.add_scalar('Loss/train', train_loss, epoch)

        if epoch % 4 == 0 or epoch == config.EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels, _ in tqdm(test_loader, desc='Test', leave=False):
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = model.forward(data)
                    loss = loss_function(predictions, labels)

                    test_loss += loss.item() / len(test_loader)
                    del data, labels
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), os.path.join(weight_dir, 'best'))
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.add_scalar('Loss/test', test_loss, epoch)
            save_metrics()
    save_metrics()