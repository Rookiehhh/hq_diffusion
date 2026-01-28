import os
import argparse
import hq_det.dataset
import numpy as np
import cv2
from tqdm import tqdm
import json


def crop_center_crop_mode(img_np, bbox, crop_size):
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    max_side = max(box_width, box_height)
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    img_h, img_w = img_np.shape[0], img_np.shape[1]
    
    if max_side <= crop_size:
        crop_half = crop_size // 2
        x1 = max(0, int(center_x - crop_half))
        y1 = max(0, int(center_y - crop_half))
        x2 = min(img_w, int(center_x + crop_half))
        y2 = min(img_h, int(center_y + crop_half))
        
        if x2 - x1 < crop_size:
            x1 = max(0, img_w - crop_size) if x1 > 0 else 0
            x2 = min(img_w, x1 + crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, img_h - crop_size) if y1 > 0 else 0
            y2 = min(img_h, y1 + crop_size)
        
        subimg = img_np[y1:y2, x1:x2, :]
        if subimg.shape[0] != crop_size or subimg.shape[1] != crop_size:
            subimg = cv2.resize(subimg, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    else:
        crop_half = max_side // 2
        x1 = max(0, int(center_x - crop_half))
        y1 = max(0, int(center_y - crop_half))
        x2 = min(img_w, int(center_x + crop_half))
        y2 = min(img_h, int(center_y + crop_half))
        
        square_size = min(x2 - x1, y2 - y1)
        center_x_actual = (x1 + x2) / 2
        center_y_actual = (y1 + y2) / 2
        x1 = max(0, int(center_x_actual - square_size / 2))
        y1 = max(0, int(center_y_actual - square_size / 2))
        x2 = min(img_w, x1 + square_size)
        y2 = min(img_h, y1 + square_size)
        
        subimg = cv2.resize(img_np[y1:y2, x1:x2, :], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    
    return subimg


def crop_original_mode(img_np, bbox, pad=50):
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    size = max(box_width, box_height)
    large_box_x = max(0, int(bbox[0] + box_width / 2 - size / 2))
    large_box_y = max(0, int(bbox[1] + box_height / 2 - size / 2))
    large_box_w = min(size, img_np.shape[1] - large_box_x)
    large_box_h = min(size, img_np.shape[0] - large_box_y)

    large_box_x = max(0, large_box_x - pad)
    large_box_y = max(0, large_box_y - pad)
    large_box_w = int(min(img_np.shape[1] - large_box_x, large_box_w + 2 * pad))
    large_box_h = int(min(img_np.shape[0] - large_box_y, large_box_h + 2 * pad))

    return img_np[large_box_y:large_box_y+large_box_h, large_box_x:large_box_x+large_box_w, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--mode", type=str, default="original", choices=["original", "center_crop"])
    parser.add_argument("--pad", type=int, default=50)
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    det_dataset = hq_det.dataset.CocoDetection(
        args.input_path,
        f"{args.input_path}/_annotations.coco.json",
        transforms=None,
    )

    output_index = 0
    output_meta = []
    for i in tqdm(range(len(det_dataset))):
        data = det_dataset[i]
        img_np = np.array(data['img'])
        bboxes = data['bboxes']
        label_names = [det_dataset.id2names[l] for l in data['cls']]

        for bbox, label_name in zip(bboxes, label_names):
            if label_name == "裂纹":
                subimg = crop_center_crop_mode(img_np, bbox, args.crop_size) if args.mode == "center_crop" else crop_original_mode(img_np, bbox, args.pad)
                
                image_filename = f"{output_index:05d}_img.jpg"
                output_index += 1
                cv2.imwrite(f"{args.output_path}/{image_filename}", cv2.cvtColor(subimg, cv2.COLOR_RGB2BGR))
                output_meta.append({"image": image_filename, "text": "defect of crack"})

    with open(f"{args.output_path}/metadata.jsonl", "w", encoding="utf-8") as f:
        for item in output_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")