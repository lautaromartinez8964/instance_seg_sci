import argparse
import json
import os.path as osp

def fix_json(path):
    print(f"[fix] {path}")
    with open(path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 1) 仅移除 background category（id=0 或 name=background）
    old_cats = coco.get('categories', [])
    new_cats = []
    for c in old_cats:
        cid = c.get('id')
        name = str(c.get('name', '')).strip().lower()
        if cid == 0 or name == 'background':
            continue
        new_cats.append(c)

    # 2) 合法类别id集合（通常应为1..15）
    valid_ids = {c['id'] for c in new_cats}

    # 3) 不重映射！只过滤非法注释（比如 category_id=0）
    old_n = len(coco.get('annotations', []))
    coco['annotations'] = [
        a for a in coco.get('annotations', [])
        if a.get('category_id') in valid_ids
    ]
    new_n = len(coco['annotations'])

    # 4) 写回 categories（保持原id和原name，不改顺序语义）
    coco['categories'] = new_cats

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(f"[done] categories={len(new_cats)} annotations {old_n}->{new_n}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    args = parser.parse_args()

    for mode in ['train', 'val']:
        p = osp.join(args.dataset_path, mode, f'instancesonly_filtered_{mode}.json')
        if osp.exists(p):
            fix_json(p)
        else:
            print(f"[skip] {p}")