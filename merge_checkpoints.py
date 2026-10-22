# pyrefly: ignore [missing-import]
import torch
import argparse

def merge_checkpoints(main_ckpt_path, text_ckpt_path, image_ckpt_path, out_path):
    print(f"Loading main checkpoint: {main_ckpt_path}")
    main_ckpt = torch.load(main_ckpt_path, map_location='cpu')
    
    # Unwrap state dict if needed
    state_dict = main_ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        if key in main_ckpt and isinstance(main_ckpt[key], dict):
            state_dict = main_ckpt[key]
            break

    if text_ckpt_path:
        print(f"Loading text-only checkpoint: {text_ckpt_path}")
        text_ckpt = torch.load(text_ckpt_path, map_location='cpu')
        t_state_dict = text_ckpt.get("model_state_dict", text_ckpt.get("state_dict", text_ckpt.get("model", text_ckpt)))
        # Assuming the final linear layer is named 'classifier' in the unimodal model
        # You may need to change 'classifier.weight' and 'classifier.bias' based on your actual unimodal architecture
        state_dict['text_only_classifier.weight'] = t_state_dict['classifier.weight']
        state_dict['text_only_classifier.bias'] = t_state_dict['classifier.bias']

    if image_ckpt_path:
        print(f"Loading image-only checkpoint: {image_ckpt_path}")
        image_ckpt = torch.load(image_ckpt_path, map_location='cpu')
        i_state_dict = image_ckpt.get("model_state_dict", image_ckpt.get("state_dict", image_ckpt.get("model", image_ckpt)))
        # Assuming the final linear layer is named 'classifier' in the unimodal model
        state_dict['image_only_classifier.weight'] = i_state_dict['classifier.weight']
        state_dict['image_only_classifier.bias'] = i_state_dict['classifier.bias']

    print(f"Saving merged checkpoint to: {out_path}")
    torch.save(state_dict, out_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge unimodal classifiers into TweFuse-W checkpoint")
    parser.add_argument('--main', type=str, required=True, help="Path to main TweFuse-W checkpoint")
    parser.add_argument('--text', type=str, help="Path to text-only model checkpoint")
    parser.add_argument('--image', type=str, help="Path to image-only model checkpoint")
    parser.add_argument('--out', type=str, default="checkpoints/twefuse_w_merged.pt", help="Output path")
    args = parser.parse_args()
    
    merge_checkpoints(args.main, args.text, args.image, args.out)
