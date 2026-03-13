好，下面我给你一套**从 0 开始、人在 `~` 目录、通过 VSCode 连 SSH 主机**就能照着做的**完整教程**。  
目标是：

1. 把 **iSAID** 下载/整理好  
2. 检查数据是否完整  
3. 配到你的 **mmdetection / instance_seg_sci** 项目里  
4. 跑通 **Mask R-CNN baseline**

我会尽量按**最稳妥、最少绕路**的方案讲。  
另外，因为你提到了 OpenDataLab / openxlab 这类工具，且这些工具和版本可能更新，我参考了官方 DSDL/OpenDataLab 文档来给你流程建议；OpenDataLab 文档明确建议先查看数据集信息/README，再进行下载和数据组织。([opendatalab.github.io](https://opendatalab.github.io/dsdl-docs/en/tutorials/dataset_download/?utm_source=openai))  
同时，iSAID 官方页面明确说明：**iSAID 的图像与 DOTA-v1.0 相同**。([captain-whu.github.io](https://captain-whu.github.io/iSAID/dataset.html?utm_source=openai))

---

# 总体思路先说清

你现在最推荐的工程组织是：

- **代码仓库**：`~/projects/mmdetection` 或你的 `instance_seg_sci`
- **数据集目录**：`~/datasets/iSAID`
- **权重目录**：`~/checkpoints`
- **实验输出**：`~/exp_dirs`

不要把原始数据直接塞进 git 仓库。

---

# Part 1：先在服务器上准备目录

你现在在 `~` 下，先执行：

```bash name=setup_dirs.sh
mkdir -p ~/projects
mkdir -p ~/datasets
mkdir -p ~/checkpoints
mkdir -p ~/exp_dirs
```

然后确认：

```bash name=check_dirs.sh
pwd
ls ~
```

---

# Part 2：确认你的代码项目位置

假设你的 mmdetection 项目在：

```text
~/projects/mmdetection
```

先进去看看：

```bash name=enter_repo.sh
cd ~/projects/mmdetection
pwd
ls
```

如果你项目名其实是 `instance_seg_sci`，那就改成：

```bash
cd ~/projects/instance_seg_sci
```

后面我先统一按 `~/projects/mmdetection` 讲，你自己替换成真实路径就行。

---

# Part 3：数据下载方案选择

## 推荐方案
**优先用 OpenDataLab / openxlab 看文件结构，再决定是否直接下载。**

原因：
- 你现在是在远程服务器
- 命令行下载更方便
- 可以先查清楚数据集里到底有没有原图、JSON、README
- OpenDataLab 文档本身就是这么建议的：先看信息/README，再组织和准备数据。([opendatalab.github.io](https://opendatalab.github.io/dsdl-docs/en/tutorials/dataset_download/?utm_source=openai))

---

# Part 4：不要在 `mmdec` 主环境里折腾下载工具

你之前已经碰到 `openxlab` 和 `SAM2` 的依赖冲突风险。  
所以我建议：

- **新建一个专门的下载环境**
- 下载完成后，训练还回 `mmdec`

## 4.1 创建下载环境

```bash name=create_download_env.sh
conda create -n odl_dl python=3.10 -y
conda activate odl_dl
python -V
```

## 4.2 安装 openxlab

```bash name=install_openxlab.sh
pip install -U pip
pip install -U openxlab
```

---

# Part 5：登录并查看 iSAID 数据集信息

## 5.1 登录
你需要先在 OpenDataLab / OpenXLab 账号里准备好 AK/SK，然后：

```bash name=openxlab_login.sh
openxlab login
```

按提示输入 AK / SK。

## 5.2 查看数据集信息

```bash name=show_isaid_info.sh
openxlab dataset info --dataset-repo OpenDataLab/iSAID
```

如果支持列文件，再试：

```bash name=list_isaid_files.sh
openxlab dataset ls --dataset-repo OpenDataLab/iSAID
```

> 有的平台命令可能叫 `ls`，有的可能不支持；如果报错也正常，你至少先拿到 `info` 和 README。

OpenDataLab 的 DSDL 文档说明，很多数据集会附带 `README.md`、`config.py`、有时还有 `tools/prepare.py`，建议用户先按 README 检查解压和组织方式。([opendatalab.github.io](https://opendatalab.github.io/dsdl-docs/en/tutorials/dataset_download/?utm_source=openai))

---

# Part 6：下载到哪里？

**下载到独立目录，不要下载到 mmdetection 仓库里。**

比如：

```bash name=prepare_isaid_dir.sh
mkdir -p ~/datasets/iSAID
```

然后下载：

```bash name=download_isaid.sh
openxlab dataset get --dataset-repo OpenDataLab/iSAID --target-path ~/datasets/iSAID
```

如果你只想先拿 README 或某些文件，也可以用定向下载命令。

---

# Part 7：下载完后怎么检查？

进入数据目录：

```bash name=enter_isaid_dir.sh
cd ~/datasets/iSAID
pwd
find . -maxdepth 3 -type d | sort | head -100
```

你重点要确认有没有这些东西：

- `train/Annotations/...json`
- `val/Annotations/...json`
- `train/images/` 或原图目录
- `val/images/` 或原图目录
- `Instance_masks/`
- `semantic_masks/`
- `README.md`

---

# Part 8：关键理解——iSAID 和 DOTA 的关系

iSAID 官方页面明确说了：  
**“The images of iSAID is the same as the DOTA-v1.0 dataset”**，也就是 iSAID 的图像和 DOTA-v1.0 相同。([captain-whu.github.io](https://captain-whu.github.io/iSAID/dataset.html?utm_source=openai))

这句话的工程含义是：

- iSAID 给你的是实例分割任务标注体系
- 原图可能就是 DOTA 那批图
- 你本地如果最终没有 `P0000.png` 这类原图，那就必须补 DOTA 图像

---

# Part 9：先检查 JSON 是否已经能直接给 MMDetection 用

你前面已经验证过一件非常好的事：

- `iSAID_train.json`
- `iSAID_train_20190823_114751.json`

里面都有：
- `images`
- `categories`
- `annotations`

这说明它们**很可能已经是 COCO 风格 JSON**，这对你特别有利。

## 9.1 再做一个标准检查脚本
在 `~/datasets/iSAID` 下新建一个脚本，比如：

```python name=tools/check_isaid_json.py
import json
from pprint import pprint

json_path = "train/Annotations/iSAID_train_20190823_114751.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("top-level keys:", data.keys())
print("num_images:", len(data["images"]))
print("num_categories:", len(data["categories"]))
print("num_annotations:", len(data["annotations"]))

print("\n== first 5 categories ==")
for c in data["categories"][:5]:
    print(c)

print("\n== first annotation keys ==")
ann = data["annotations"][0]
print(ann.keys())
pprint(ann)
```

运行：

```bash name=run_check_isaid_json.sh
cd ~/datasets/iSAID
python tools/check_isaid_json.py
```

你要看：
- `categories` 是否正常
- `annotations` 里是否有 `bbox`, `category_id`, `segmentation`, `area`, `iscrowd`

如果都有，那基本可以直接走 `CocoDataset`。

---

# Part 10：检查原图是否存在

这是最关键的。

```python name=tools/check_raw_images_exist.py
import json
import os

json_path = "train/Annotations/iSAID_train_20190823_114751.json"
img_dir = "train/images"   # 这里后面按你的真实目录改

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

missing = []
for item in data["images"][:100]:
    fp = os.path.join(img_dir, item["file_name"])
    if not os.path.exists(fp):
        missing.append(fp)

print("checked:", 100)
print("missing:", len(missing))
for x in missing[:20]:
    print(x)
```

运行：

```bash name=run_check_raw_images.sh
cd ~/datasets/iSAID
python tools/check_raw_images_exist.py
```

## 结果怎么解释？
- 如果 `missing = 0`：很好，说明你已有原图
- 如果缺很多：你就要补 DOTA-v1.0 图像

---

# Part 11：如果没有原图，怎么办？

那就补 DOTA 图像。  
因为 iSAID 官方已经明确说明图像来自 DOTA-v1.0。([captain-whu.github.io](https://captain-whu.github.io/iSAID/dataset.html?utm_source=openai))

你最终需要把训练时的图像整理成类似：

```text
~/datasets/iSAID/
├── train/
│   ├── images/
│   └── Annotations/
├── val/
│   ├── images/
│   └── Annotations/
```

也就是说，不管原图最初来自哪里，最后都建议你整理成**iSAID 自己的 train/val 目录结构**，方便 MMDetection 配置。

---

# Part 12：和你的 mmdetection 项目连接

这一部分最关键。

假设你的最终目录是：

```text
~/datasets/iSAID/
├── train/images/
├── train/Annotations/iSAID_train_20190823_114751.json
├── val/images/
└── val/Annotations/iSAID_val_*.json
```

那么你在 `mmdetection` 里要做两件事：

---

## 12.1 新建数据集配置文件

在项目里新建：

```text
configs/_base_/datasets/isaid_instance.py
```

内容：

```python name=configs/_base_/datasets/isaid_instance.py
dataset_type = 'CocoDataset'
data_root = '/home/你的用户名/datasets/iSAID/'

metainfo = dict(
    classes=(
        'plane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
        'basketball_court', 'ground_track_field', 'harbor', 'bridge',
        'large_vehicle', 'small_vehicle', 'helicopter', 'roundabout',
        'soccer_ball_field', 'swimming_pool'
    )
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/Annotations/iSAID_train_20190823_114751.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/Annotations/iSAID_val_20190823_*.json',  # 后面改成真实文件名
        data_prefix=dict(img='val/images/')
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/Annotations/iSAID_val_20190823_*.json',
    metric=['bbox', 'segm']
)

test_evaluator = val_evaluator
```

> 注意：`val` 的 json 文件名你要填成真实值。  
> 还有 `classes` 必须和 `categories` 里名称顺序严格一致。

---

## 12.2 继承一个 Mask R-CNN baseline config

比如基于官方的 `mask_rcnn_r50_fpn_1x_coco.py`，新建：

```text
configs/mask_rcnn/mask_rcnn_r50_fpn_isaid.py
```

内容示例：

```python name=configs/mask_rcnn/mask_rcnn_r50_fpn_isaid.py
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/isaid_instance.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)
)

work_dir = './work_dirs/mask_rcnn_r50_fpn_isaid'
```

---

# Part 13：先做“数据可视化检查”，再训练

这一步别省，非常重要。

MMDetection 一般有浏览数据集的工具。你可以先运行类似：

```bash name=browse_dataset.sh
cd ~/projects/mmdetection
python tools/analysis_tools/browse_dataset.py configs/mask_rcnn/mask_rcnn_r50_fpn_isaid.py --output-dir work_dirs/browse_isaid
```

如果你仓库工具路径略有不同，就 `ls tools` 看一下。

你要看的是：
- 图片是否能打开
- bbox 是否对齐
- mask 是否对齐
- 类别名称是否正确

---

# Part 14：开始训练 baseline

确认可视化没问题后：

```bash name=train_isaid_baseline.sh
cd ~/projects/mmdetection
conda activate mmdec
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_isaid.py
```

如果你是多卡，再换成分布式脚本。

---

# Part 15：训练前的最小冒烟测试

不要一上来全量训练。  
建议先做一个“冒烟版”：

- 把 train json 临时裁成前 100 张图
- val 裁成前 20 张
- 跑 1 epoch

这样你能很快发现问题：
- 路径错
- 图片缺失
- 类别不匹配
- segmentation 读不了
- 显存不够

---

# Part 16：你现在应该怎么执行，按顺序来

我给你一个**今晚/现在就能照做的清单**：

## 第一步：准备下载环境
```bash
conda create -n odl_dl python=3.10 -y
conda activate odl_dl
pip install -U openxlab
```

## 第二步：登录并查看 iSAID
```bash
openxlab login
openxlab dataset info --dataset-repo OpenDataLab/iSAID
```

## 第三步：下载到独立目录
```bash
mkdir -p ~/datasets/iSAID
openxlab dataset get --dataset-repo OpenDataLab/iSAID --target-path ~/datasets/iSAID
```

## 第四步：检查目录结构
```bash
cd ~/datasets/iSAID
find . -maxdepth 3 -type d | sort | head -100
find . -maxdepth 3 -type f | head -100
```

## 第五步：检查 JSON 和原图
运行我上面给你的两个 Python 检查脚本。

## 第六步：如果原图缺失，补 DOTA 图像
然后把原图整理到：
- `train/images/`
- `val/images/`

## 第七步：在 mmdetection 里写 dataset config
- `configs/_base_/datasets/isaid_instance.py`
- `configs/mask_rcnn/mask_rcnn_r50_fpn_isaid.py`

## 第八步：先 browse_dataset
```bash
python tools/analysis_tools/browse_dataset.py ...
```

## 第九步：再 train baseline
```bash
python tools/train.py ...
```

---

# Part 17：给你的科研建议

你现在最重要的不是“立刻把全量 iSAID 训到最好”，而是：

1. **今天把数据链路打通**
2. **明天拿到第一个可训练 baseline**
3. **后天开始做子集实验和蒸馏接入**

你已经把 SAM2 环节跑通了，这非常强。  
现在只要把 iSAID 配好，你的研究就正式进入“有实验产出”的阶段了。

---

# 你接下来最适合做的事

现在请你按这个顺序给我回：

1. `openxlab dataset info --dataset-repo OpenDataLab/iSAID` 输出  
2. 下载后 `find ~/datasets/iSAID -maxdepth 3 -type d | sort | head -100` 输出  
3. 你的项目真实路径到底是  
   - `~/projects/mmdetection`
   - 还是 `~/projects/instance_seg_sci`

然后我下一条就**按你的真实目录**，直接给你写一版可以复制的：
- `isaid_instance.py`
- `mask_rcnn_r50_fpn_isaid.py`

让你直接开跑。