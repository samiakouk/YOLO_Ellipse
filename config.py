import os
import torchvision.transforms as T


BATCH_SIZE = 64
EPOCHS = 80
WARMUP_EPOCHS = 20

C = 20
B = 2
S = 7

EPSILON = 1e-6
IMAGE_SIZE = (448, 448)

S = 7       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 20      # Number of classes in the dataset
