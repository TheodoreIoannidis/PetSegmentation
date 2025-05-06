# main.py
import argparse
from train import train
from unsupervised import run_unsupervised  
from inception import InceptionSegment
from unet import UNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['unet', 'inception'], default='unet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_dir', type=str, default='./data/images')
    parser.add_argument('--mask_dir', type=str, default='./data/masks')
    args = parser.parse_args()

    if args.model == 'unet':
        model = UNet(num_classes=2)
    elif args.model == 'inception':
        model = InceptionSegment(num_classes=2)

    # Call train loop from train.py
    train(
        model=model,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    run_unsupervised(model_name='gmm', num_samples=20)

if __name__ == '__main__':
    main()
