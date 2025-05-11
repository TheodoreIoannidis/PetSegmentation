# main.py
import argparse
from train import *
from unsupervised import run_unsupervised  
from inception import InceptionSegment
from unet import UNet

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Segmentation Experiments")
    parser.add_argument('--mode', type=str, choices=['supervised', 'unsupervised'], default='supervised')
    
    parser.add_argument('--model', type=str, choices=['unet', 'inception'], default='unet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    
    parser.add_argument('--img_dir', type=str, default='./data/images')
    parser.add_argument('--mask_dir', type=str, default='./data/masks')
    args = parser.parse_args()

    if args.mode == 'supervised':
        if args.model == 'unet':
            model = UNet(num_classes=2)
        elif args.model == 'inception':
            model = InceptionSegment(num_classes=2)

        train(
            model=model,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            # visualize=True
        )

    elif args.mode == 'unsupervised':
        run_unsupervised(
            model_name=args.unsupervised_model,
            image_dir=args.img_dir,
            mask_dir=args.mask_dir,
            num_samples=args.num_samples,
            # visualize=True,
            visualize=args.visualize
        )

if __name__ == '__main__':
    main()
