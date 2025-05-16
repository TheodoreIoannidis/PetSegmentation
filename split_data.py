import os
import random

random.seed(42)

image_dir = './data/images'
all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(all_images)

n = len(all_images)
train_idx = int(0.7 * n)
val_idx = int(0.85 * n)

train = all_images[:train_idx]
val = all_images[train_idx:val_idx]
test = all_images[val_idx:]

os.makedirs('./splits', exist_ok=True)
for name, data in [('train.txt', train), ('val.txt', val), ('test.txt', test)]:
    with open(f'./splits/{name}', 'w') as f:
        for filename in data:
            f.write(filename + '\n')
