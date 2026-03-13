import os, json, cv2, numpy as np
from pathlib import Path

def create_dummy_rs_dataset():
    data_root = Path('data/dummy_rs')
    (data_root / 'train').mkdir(parents=True, exist_ok=True)
    (data_root / 'annotations').mkdir(exist_ok=True)

    coco_format = {
        "images":[], "annotations": [],
        "categories":[
            {"id": 1, "name": "ship", "supercategory": "rs"},
            {"id": 2, "name": "tank", "supercategory": "rs"}
        ]
    }

    anno_id = 1
    # 生成 50 张 512x512 的模拟遥感图
    for img_id in range(1, 51):
        img = np.ones((512, 512, 3), dtype=np.uint8) * 100 # 灰色海洋/陆地背景
        
        # 随机画一个 tank (银色圆形)
        cx, cy = np.random.randint(50, 450, 2)
        r = np.random.randint(20, 40)
        cv2.circle(img, (cx, cy), r, (150, 150, 150), -1) 
        
        # 随机画一个 ship (红色矩形)
        sx, sy = np.random.randint(50, 400, 2)
        sw, sh = np.random.randint(20, 40), np.random.randint(60, 100)
        cv2.rectangle(img, (sx, sy), (sx+sw, sy+sh), (50, 50, 200), -1) 
        
        cv2.imwrite(str(data_root / f'train/{img_id:04d}.jpg'), img)
        coco_format["images"].append({"id": img_id, "file_name": f"{img_id:04d}.jpg", "width": 512, "height": 512})
        
        # 写入 Mask 标注信息
        coco_format["annotations"].append({
            "id": anno_id, "image_id": img_id, "category_id": 2,
            "bbox":[int(cx-r), int(cy-r), int(r*2), int(r*2)],
            "area": float(np.pi * r * r), "iscrowd": 0,
            "segmentation": [[int(cx-r), int(cy), int(cx), int(cy-r), int(cx+r), int(cy), int(cx), int(cy+r)]]
        })
        anno_id += 1
        
        coco_format["annotations"].append({
            "id": anno_id, "image_id": img_id, "category_id": 1,
            "bbox":[int(sx), int(sy), int(sw), int(sh)],
            "area": float(sw * sh), "iscrowd": 0,
            "segmentation": [[int(sx), int(sy), int(sx+sw), int(sy), int(sx+sw), int(sy+sh), int(sx), int(sy+sh)]]
        })
        anno_id += 1

    with open(data_root / 'annotations/train.json', 'w') as f:
        json.dump(coco_format, f)
    print("✅ 模拟遥感数据集生成成功！存放于: data/dummy_rs/")

if __name__ == '__main__':
    create_dummy_rs_dataset()