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

    parser.add_argument('--unsupervised_model', type=str, choices=['kmeans', 'gmm'], default='kmeans')
    parser.add_argument('--split_file', type=str, default='./splits/test.txt')
    parser.add_argument('--visualize', type=str, default='False')
    parser.add_argument('--post', type=str, choices=['none', 'open', 'close', 'erosion', 'dilation'], default='none',
                    help='Optional postprocessing operation to apply to predicted masks.')

    args = parser.parse_args()

    if args.mode == 'supervised':
        if args.model == 'unet':
            model = UNet(num_classes=2)
        elif args.model == 'inception':
            model = InceptionSegment(num_classes=2)
        
        try:
            model.load_state_dict(torch.load(f"./{args.model}_model.pth"))
            print("Model loaded successfully.")
        except Exception as e:
            train(
                model=model,
                img_dir=args.img_dir,
                mask_dir=args.mask_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
         
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        test_dataset = ISICDataset(args.img_dir, args.mask_dir, transform, file_list='./splits/test.txt')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        pred_masks, gt_masks, images = test_model(model, test_loader)

    elif args.mode == 'unsupervised':
        pred_masks, gt_masks, images = run_unsupervised(
            model_name=args.unsupervised_model,
            image_dir=args.img_dir,
            mask_dir=args.mask_dir,
            file_list=args.split_file,
        )

    print("\nRaw Mask results")
    evaluate_masks(pred_masks, gt_masks)
    if args.post != 'none':
        pred_masks = postprocess(pred_masks, mode=args.post)
        print("\nPostprocessed Mask results")
        evaluate_masks(pred_masks, gt_masks)

    if args.visualize != 'False':
        idx = np.random.randint(0, len(images))
        visualize_overlay(images[idx], gt_masks[idx], pred_masks[idx], alpha=0.5)
        plt.close()

if __name__ == '__main__':
    main()
