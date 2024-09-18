"""

@author:  AIcandy 
@website: aicandy.vn

"""

from torchvision import transforms
from aicandy_utils_src_lkkqrsdm.aicandy_ssd300_utils_xslstyan import *
from PIL import Image, ImageDraw, ImageFont

# python aicandy_ssd300_test_qgtlqrlv.py --image_path image_test.jpg --checkpoint aicandy_model_out_gnloibxd/aicandy_checkpoint_ssd300.pth


def predict(image_path, checkpoint, min_score, max_overlap, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model checkpoint
    checkpoint = torch.load(checkpoint, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    original_image = Image.open(image_path, mode='r')
    original_image = original_image.convert('RGB')

    # Transform
    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    
    # Move tensors to the same device as the model
    predicted_locs = predicted_locs.to(device)
    predicted_scores = predicted_scores.to(device)
    
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap)
    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels == ['background']:
        print("No objects found in the picture")
    else:        # Annotate
        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("aicandy_utils_src_lkkqrsdm/arial.ttf", 15)

        for i in range(det_boxes.size(0)):
            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[det_labels[i]])

            # Text
            text = det_labels[i].upper()
            text_bbox = draw.textbbox((0, 0), text, font=font) 
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_location = [box_location[0] + 2., box_location[1] - text_height]
            textbox_location = [box_location[0], box_location[1] - text_height, box_location[0] + text_width + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=text, fill='white', font=font)

        del draw
        image_out_path = os.path.join(output_dir, 'image_output.jpg')
        annotated_image.save(image_out_path) 
        print(f"The results have been saved in: ", image_out_path)


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint')

    args = parser.parse_args()
    image_out_path = predict(args.image_path, args.checkpoint, 0.2, 0.5, 'output')
    