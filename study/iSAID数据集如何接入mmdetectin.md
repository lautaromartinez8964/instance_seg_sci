太好了，现在信息齐了。  
我直接给你一套**“原始 iSAID → patch 版 iSAID_patches → 接入 mmdetection”完整命令流**，尽量按你当前机器和目录来写。

而且这条流程和两边是一致的：

- **官方 iSAID_Devkit**：要求把 train/val/test 原始图像切成 patch，再生成 COCO 风格 patch 标注。官方 README 明确给了流程：`split.py --set train,val`、`split.py --set test`、`preprocess.py --set train,val`，并要求最终得到 `iSAID_patches` 结构。([github.com](https://github.com/CAPTAIN-WHU/iSAID_Devkit?utm_source=openai))
- **你自己的仓库**：已经内置 `iSAIDDataset`、`configs/_base_/datasets/isaid_instance.py`、`projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py`，并明确要求先按 devkit 做 patch 预处理，再运行 `projects/iSAID/isaid_json.py`。

另外，你贴的这句配置：

```python
dataset_type = 'iSAIDDataset'
data_root = 'data/iSAID/'
...
```

也再次说明：  
**你项目这条线默认期待的是处理后的 `data/iSAID/` patch 数据，而不是 OpenDataLab 的 `raw/` 大图目录。** 

---

# 一、你当前真实目录情况

你的原始数据在：

```bash
/home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw
```

并且已经有：

- `train/Annotations`
- `train/Instance_masks`
- `train/Semantic_masks`
- `train/images`
- `val/Annotations`
- `val/Instance_masks`
- `val/Semantic_masks`
- `val/images`
- `test`

这说明：

## 结论
你已经具备**做 patch 预处理**的原始输入了。  
下一步不再是“检查原始数据”，而是**标准化整理 + 用 devkit 切 patch**。

---

# 二、总流程图

你接下来要走的是：

```text
OpenDataLab___iSAID/raw
        ↓
整理成 devkit 期待的 iSAID 原始结构
        ↓
iSAID_Devkit split.py 切 patch
        ↓
iSAID_Devkit preprocess.py 生成 patch 标注 json
        ↓
得到 iSAID_patches/
        ↓
运行 projects/iSAID/isaid_json.py
        ↓
软链接到 mmdetection/data/iSAID
        ↓
跑 Mask R-CNN baseline
```

---

# 三、完整命令流

下面默认：

- 数据根目录：`/home/yxy18034962/datasets/iSAID`
- 项目根目录：`/home/yxy18034962/projects/mmdetection`

你基本直接复制就能用。

---

## Step 0：进入一个干净位置

```bash name=go_dataset_root.sh
cd /home/yxy18034962/datasets/iSAID
pwd
ls
```

---

## Step 1：整理出 devkit 期待的“原始 iSAID”目录

官方 devkit 期待的输入结构不是 OpenDataLab 的 `raw/train/images/images` 这种，而是更干净的：

```text
iSAID/
├── train/images/
├── val/images/
└── test/images/
```

并且 train/val 的 `images/` 目录里要同时有：
- 原图 `P0002.png`
- 实例 mask `P0002_instance_id_RGB.png`
- 语义 mask `P0002_instance_color_RGB.png`

官方 README 就是这么写的。([github.com](https://github.com/CAPTAIN-WHU/iSAID_Devkit?utm_source=openai))

### 1.1 新建标准原始目录

```bash name=make_clean_isaid_raw.sh
mkdir -p /home/yxy18034962/datasets/iSAID/iSAID_raw/train/images
mkdir -p /home/yxy18034962/datasets/iSAID/iSAID_raw/val/images
mkdir -p /home/yxy18034962/datasets/iSAID/iSAID_raw/test/images
```

### 1.2 把 train 原图拷进去
你说解压后多了一层 `images/`，所以原图路径应是：

```text
/home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw/train/images/images
```

执行：

```bash name=copy_train_images.sh
cp /home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw/train/images/images/*.png \
   /home/yxy18034962/datasets/iSAID/iSAID_raw/train/images/
```

### 1.3 把 train instance masks 拷进去
先确认解压后 mask 文件实际在哪一层，如果还是 `images/` 子目录，就用下面这个；如果不是，你自己 `ls` 一下改成真实路径。

```bash name=copy_train_instance_masks.sh
cp /home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw/train/Instance_masks/images/*.png \
   /home/yxy18034962/datasets/iSAID/iSAID_raw/train/images/
```

如果解压后是 `.../Instance_masks/images/images/*.png`，就改那一层。

### 1.4 把 train semantic masks 拷进去

```bash name=copy_train_semantic_masks.sh
cp /home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw/train/Semantic_masks/images/*.png \
   /home/yxy18034962/datasets/iSAID/iSAID_raw/train/images/
```

### 1.5 处理 val
同理：

```bash name=copy_val_images_and_masks.sh
cp /home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw/val/images/images/*.png \
   /home/yxy18034962/datasets/iSAID/iSAID_raw/val/images/

cp /home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw/val/Instance_masks/images/*.png \
   /home/yxy18034962/datasets/iSAID/iSAID_raw/val/images/

cp /home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw/val/Semantic_masks/images/*.png \
   /home/yxy18034962/datasets/iSAID/iSAID_raw/val/images/
```

### 1.6 处理 test
test 只需要原图，不需要 mask。官方也明确说 test mask withheld。([github.com](https://github.com/CAPTAIN-WHU/iSAID_Devkit?utm_source=openai))

```bash name=copy_test_images.sh
cp /home/yxy18034962/datasets/iSAID/OpenDataLab___iSAID/raw/test/images/images/*.png \
   /home/yxy18034962/datasets/iSAID/iSAID_raw/test/images/
```

如果 test 还没解压，先去 test 目录把 zip 解压了再执行。

---

## Step 2：快速检查整理后的原始目录

```bash name=check_clean_raw.sh
find /home/yxy18034962/datasets/iSAID/iSAID_raw -maxdepth 3 -type d | sort
find /home/yxy18034962/datasets/iSAID/iSAID_raw/train/images -name "*.png" | head
find /home/yxy18034962/datasets/iSAID/iSAID_raw/val/images -name "*.png" | head
```

你应该能看到 train/val 下同一目录里同时有：
- `P000x.png`
- `P000x_instance_id_RGB.png`
- `P000x_instance_color_RGB.png`

---

## Step 3：克隆官方 iSAID_Devkit

```bash name=clone_isaid_devkit.sh
cd /home/yxy18034962/datasets/iSAID
git clone https://github.com/CAPTAIN-WHU/iSAID_Devkit.git
```

---

## Step 4：创建 devkit 环境
官方 README 给了环境说明，包括 `environment.yml`、pycocotools、cityscapesScripts 等。([github.com](https://github.com/CAPTAIN-WHU/iSAID_Devkit?utm_source=openai))  
但你未必要完全照它��套老环境。先尝试一个简单环境；如果报依赖错，再回补。

```bash name=create_isaid_devkit_env.sh
conda create -n isaid_devkit python=3.8 -y
conda activate isaid_devkit
pip install -U pip setuptools wheel
pip install numpy pillow opencv-python tqdm shapely pycocotools matplotlib
```

如果 devkit 后面报缺包，我们再补。

---

## Step 5：按官方方式把数据挂到 devkit 里

官方 README 写的是在 `preprocess/` 目录里给数据建软链接到 `./dataset/`。([github.com](https://github.com/CAPTAIN-WHU/iSAID_Devkit?utm_source=openai))

```bash name=link_dataset_into_devkit.sh
cd /home/yxy18034962/datasets/iSAID/iSAID_Devkit/preprocess
ln -s /home/yxy18034962/datasets/iSAID/iSAID_raw ./dataset
ls -l
```

这里的 `dataset` 链接最终应指向你的 `iSAID_raw`。

---

## Step 6：切 train/val patch

官方命令：

```bash
python split.py --set train,val
```

来源于 devkit README。([github.com](https://github.com/CAPTAIN-WHU/iSAID_Devkit?utm_source=openai))

所以执行：

```bash name=split_train_val.sh
cd /home/yxy18034962/datasets/iSAID/iSAID_Devkit/preprocess
python split.py --set train,val
```

### 然后切 test

```bash name=split_test.sh
cd /home/yxy18034962/datasets/iSAID/iSAID_Devkit/preprocess
python split.py --set test
```

---

## Step 7：生成 patch 版 COCO 标注

官方命令：

```bash
python preprocess.py --set train,val
```

来源于 devkit README。([github.com](https://github.com/CAPTAIN-WHU/iSAID_Devkit?utm_source=openai))

执行：

```bash name=generate_patch_json.sh
cd /home/yxy18034962/datasets/iSAID/iSAID_Devkit/preprocess
python preprocess.py --set train,val
```

---

## Step 8：确认输出的 `iSAID_patches` 在哪
官方 README 说最终会得到 `iSAID_patches/`，里面包括：

- `train/images/...patch.png`
- `val/images/...patch.png`
- `test/images/...patch.png`
- `instance_only_filtered_train.json`
- `instance_only_filtered_val.json`。([github.com](https://github.com/CAPTAIN-WHU/iSAID_Devkit?utm_source=openai))

你现在执行：

```bash name=find_patched_output.sh
cd /home/yxy18034962/datasets/iSAID/iSAID_Devkit
find . -maxdepth 3 -type d | grep -E "patch|iSAID_patches|train|val|test" | sort
find . -name "*filtered*.json"
```

---

# 四、和你仓库的命名对齐

这里要注意一个小坑：

## 官方 README 里有时写 `instance_only_filtered_train.json`
但你仓库的：
- `configs/_base_/datasets/isaid_instance.py`
- `projects/iSAID/isaid_json.py`

期待的是：

- `instancesonly_filtered_train.json`
- `instancesonly_filtered_val.json`。

所以如果 devkit 产出的文件名是 `instance_only_filtered_train.json`，你要**重命名**成仓库期待的名字。

### 假设 patch 输出在：
`/home/yxy18034962/datasets/iSAID/iSAID_patches`

执行：

```bash name=rename_patch_jsons.sh
cd /home/yxy18034962/datasets/iSAID/iSAID_patches

if [ -f train/instance_only_filtered_train.json ]; then
  mv train/instance_only_filtered_train.json train/instancesonly_filtered_train.json
fi

if [ -f val/instance_only_filtered_val.json ]; then
  mv val/instance_only_filtered_val.json val/instancesonly_filtered_val.json
fi
```

---

# 五、运行你仓库自带的 json 修正脚本

你仓库这个脚本会给 categories 加一个 `background`。

```bash name=run_repo_isaid_json_fix.sh
cd /home/yxy18034962/projects/mmdetection
conda activate mmdec
python projects/iSAID/isaid_json.py /home/yxy18034962/datasets/iSAID/iSAID_patches
```

---

# 六、接入 mmdetection

你仓库配置默认：

```python
data_root = 'data/iSAID/'
```

所以最方便的是直接做软链接。

```bash name=link_patches_to_mmdet.sh
cd /home/yxy18034962/projects/mmdetection
mkdir -p data
ln -s /home/yxy18034962/datasets/iSAID/iSAID_patches data/iSAID
ls -l data
```

如果 `data/iSAID` 已存在，先删掉旧链接：

```bash
rm -rf /home/yxy18034962/projects/mmdetection/data/iSAID
```

然后再 `ln -s`。

---

# 七、先做数据可视化检查

你仓库有标准的 iSAID 数据配置和 baseline 配置，所以现在先不要急训练，先 browse。  
数据配置：

```python name=configs/_base_/datasets/isaid_instance.py url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/2242151f31d8811a2e3e9f55699b99fe80a7d5a4/configs/_base_/datasets/isaid_instance.py#L1-L26
dataset_type = 'iSAIDDataset'
data_root = 'data/iSAID/'
...
```

baseline 配置：

```python name=projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/2242151f31d8811a2e3e9f55699b99fe80a7d5a4/projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py#L1-L6
_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/isaid_instance.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]
```

所以先执行：

```bash name=browse_isaid_dataset.sh
cd /home/yxy18034962/projects/mmdetection
conda activate mmdec
python tools/analysis_tools/browse_dataset.py \
    projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py \
    --output-dir work_dirs/browse_isaid
```

如果你仓库里 `browse_dataset.py` 路径不同，就 `find tools -name browse_dataset.py` 看一下。

---

# 八、开始 baseline 冒烟训练

仓库 README 已经给了训练命令：

```bash name=train_isaid_baseline.sh
cd /home/yxy18034962/projects/mmdetection
conda activate mmdec
python tools/train.py projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py
```

---

# 九、我建议你先做“2 epoch 冒烟”而不是直接全量 12 epoch

虽然默认是 `schedule_1x`，但为了先确认数据和 loss 没问题，你可以先临时复制一个 config 做冒烟版。

新建一个文件，比如：

```python name=projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid_smoke.py
_base_ = './mask_rcnn_r50_fpn_1x_isaid.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2)
)
```

然后跑：

```bash name=train_isaid_smoke.sh
cd /home/yxy18034962/projects/mmdetection
conda activate mmdec
python tools/train.py projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid_smoke.py
```

---

# 十、你最可能遇到的 5 个坑

## 1. devkit 期望的输入目录不对
如果 `split.py` 报找不到图，通常是 `iSAID_raw/train/images/` 里没有同时放：
- 原图
- instance mask
- semantic mask

## 2. OpenDataLab 解压多套了一层 `images/`
你已经碰到了，所以凡是 `cp .../*.png` 前，都先 `ls` 看实际层级。

## 3. patch 输出 json 文件名和仓库期望不一致
官方文档里有 `instance_only_filtered_*.json` 的写法，但你仓库期望 `instancesonly_filtered_*.json`。  
如果不一致，就按我上面给你的 `mv` 重命名。

## 4. `projects/iSAID/isaid_json.py` 找不到 json
这通常是因为 patch 目录没放对、文件名没改对。  
脚本源码明确写死在找：  
- `train/instancesonly_filtered_train.json`
- `val/instancesonly_filtered_val.json`。

## 5. 类别顺序疑惑
你原始 JSON 的类别顺序和仓库 `iSAIDDataset.METAINFO` 可能不完全同字面，但仓库已经专门为 iSAID 写了 dataset class，优先相信仓库这条线。

---

# 十一、给你一个最短执行清单

如果��想最省脑子，今天就照这个顺序：

```bash
# 1) 整理成 iSAID_raw/train|val|test/images
# 2) clone devkit
# 3) devkit/preprocess 里 ln -s iSAID_raw ./dataset
# 4) python split.py --set train,val
# 5) python split.py --set test
# 6) python preprocess.py --set train,val
# 7) 如有需要，重命名 instance_only_filtered_*.json -> instancesonly_filtered_*.json
# 8) python projects/iSAID/isaid_json.py /home/yxy18034962/datasets/iSAID/iSAID_patches
# 9) ln -s /home/yxy18034962/datasets/iSAID/iSAID_patches /home/yxy18034962/projects/mmdetection/data/iSAID
# 10) python tools/analysis_tools/browse_dataset.py ...
# 11) python tools/train.py projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py
```

---

# 十二、我给你的科研建议

你现在已经完成：
- SAM2 跑通
- 中间特征拿到
- iSAID 原始数据拿到并验证

这说明你��“真正开做论文实验”就差这一步 patch 预处理。  
而这个步骤一旦通了，你后面就能并行推进：

1. **ResNet50 + Mask R-CNN baseline**
2. **VMamba/LightMamba 替换 backbone**
3. **SAM 特征蒸馏接入**

所以今天你的核心目标不是“调高 AP”，而是：

> **把 iSAID 数据链路正式打通。**

---

如果你愿意，下一条你把这几个命令的输出贴给我：

1. `ls /home/yxy18034962/datasets/iSAID/iSAID_raw/train/images | head`
2. `ls /home/yxy18034962/datasets/iSAID/iSAID_Devkit/preprocess`
3. `python split.py --set train,val` 的输出
4. `find /home/yxy18034962/datasets/iSAID -name "*filtered*.json"`

我可以继续像导师一样陪你，把这一步彻底走通。