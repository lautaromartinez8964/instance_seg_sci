import argparse
import json
import os
import os.path as osp

# 必须与 mmdet/datasets/isaid.py 中 METAINFO['classes'] 完全一致（含 background）
METAINFO_CLASSES = (
    'background', 'ship', 'store_tank', 'baseball_diamond',
    'tennis_court', 'basketball_court', 'Ground_Track_Field',
    'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
    'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
    'Harbor'
)

# 规范化别名（统一到 METAINFO_CLASSES 的精确写法）
NAME_ALIAS = {
    'storage_tank': 'store_tank',
    'store_tank': 'store_tank',
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
    'background': 'background',
}

STD_NAME2ID = {name: i for i, name in enumerate(METAINFO_CLASSES)}  # 0..15

def canon_name(name: str) -> str:
    name = str(name).strip()
    key = name.lower()
    if key in NAME_ALIAS:
        return NAME_ALIAS[key]
    # 大小写兜底匹配到标准类名
    for std in METAINFO_CLASSES:
        if std.lower() == key:
            return std
    return name

def rewrite_one_json(json_path: str):
    print(f"[rewrite] {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 1) old_id -> old_name(规范化)
    old_id_to_cname = {}
    for c in coco.get('categories', []):
        old_id = c.get('id')
        cname = canon_name(c.get('name', ''))
        if old_id is None:
            continue
        old_id_to_cname[old_id] = cname

    # 2) old_id -> new_id(0..15) （按“名字”映射到标准 id）
    old_id_to_new_id = {}
    for old_id, cname in old_id_to_cname.items():
        if cname in STD_NAME2ID:
            old_id_to_new_id[old_id] = STD_NAME2ID[cname]

    # 3) 重写 annotations：category_id 必须落在 1..15（不允许 0=background）
    new_anns = []
    dropped = 0
    for ann in coco.get('annotations', []):
        old = ann.get('category_id', None)
        if old not in old_id_to_new_id:
            dropped += 1
            continue
        new_id = old_id_to_new_id[old]
        if new_id == 0:
            # 严格禁止 annotation 使用 background
            dropped += 1
            continue
        ann['category_id'] = new_id
        new_anns.append(ann)

    coco['annotations'] = new_anns

    # 4) 重写 categories：固定为 0..15 且顺序与 METAINFO_CLASSES 一致
    coco['categories'] = [{'id': i, 'name': name} for i, name in enumerate(METAINFO_CLASSES)]

    # 5) 原子写回，避免 JSON 写坏
    tmp = json_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(coco, f)
    os.replace(tmp, json_path)

    print(f"[done] cats={len(coco['categories'])} anns={len(new_anns)} dropped={dropped}")

def main():
    parser = argparse.ArgumentParser(description='Rewrite iSAID coco json to match iSAIDDataset.METAINFO (with background)')
    parser.add_argument('dataset_path', help='iSAID_patches folder path')
    args = parser.parse_args()

    for mode in ['train', 'val']:
        p = osp.join(args.dataset_path, mode, f'instancesonly_filtered_{mode}.json')
        if osp.exists(p):
            rewrite_one_json(p)
        else:
            print(f"[skip] not found: {p}")

if __name__ == '__main__':
    main()