import argparse
import json
import os.path as osp

# MMDetection iSAIDDataset 需要的严格类别列表和顺序（连大小写都一模一样）
METAINFO_CLASSES = (
    'ship', 'store_tank', 'baseball_diamond',
    'tennis_court', 'basketball_court', 'Ground_Track_Field',
    'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
    'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
    'Harbor'
)

# iSAID原始名称 到 METAINFO名称的翻译字典
NAME_ALIAS = {
    'storage_tank': 'store_tank',
    'ground_track_field': 'Ground_Track_Field',
    'bridge': 'Bridge',
    'large_vehicle': 'Large_Vehicle',
    'small_vehicle': 'Small_Vehicle',
    'helicopter': 'Helicopter',
    'swimming_pool': 'Swimming_pool',
    'roundabout': 'Roundabout',
    'soccer_ball_field': 'Soccer_ball_field',
    'harbor': 'Harbor',
    'plane': 'plane',
    'ship': 'ship',
    'baseball_diamond': 'baseball_diamond',
    'tennis_court': 'tennis_court',
    'basketball_court': 'basketball_court',
    'store_tank': 'store_tank'
}

def json_convert(path):
    print(f"正在进行深度洗数据与对齐: {path}")
    with open(path, 'r') as f:
        coco_data = json.load(f)
        
    # 1. 记住原来的 id 和名字
    old_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 2. 建立 原id -> 新id 的精确映射
    old_id_to_new_id = {}
    for old_id, old_name in old_id_to_name.items():
        lower_name = old_name.lower()
        if lower_name in NAME_ALIAS:
            canonical_name = NAME_ALIAS[lower_name]
        else:
            canonical_name = old_name
            for cls in METAINFO_CLASSES:
                if cls.lower() == old_name.lower():
                    canonical_name = cls
                    break
        
        # 获取该类别在 METAINFO 中的标准索引
        if canonical_name in METAINFO_CLASSES:
            new_id = METAINFO_CLASSES.index(canonical_name)
            old_id_to_new_id[old_id] = new_id
            
    # 3. 核心：把几十万个框的 category_id 全部洗掉重写！
    for ann in coco_data['annotations']:
        old_id = ann['category_id']
        if old_id in old_id_to_new_id:
            ann['category_id'] = old_id_to_new_id[old_id]
            
    # 4. 最后塞入完美字典
    new_categories = []
    for idx, name in enumerate(METAINFO_CLASSES):
        new_categories.append({'id': idx, 'name': name})
    coco_data['categories'] = new_categories
    
    # 保存
    with open(path, 'w') as f:
        json.dump(coco_data, f)
    print("完美对齐成功！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert iSAID dataset to mmdetection format')
    parser.add_argument('dataset_path', help='iSAID folder path')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    
    for dataset_mode in ['train', 'val']:
        json_file = 'instancesonly_filtered_' + dataset_mode + '.json'
        json_file_path = osp.join(dataset_path, dataset_mode, json_file)
        
        if osp.exists(json_file_path):
            json_convert(json_file_path)
        else:
            print(f"跳过: 未找到 {json_file_path}")