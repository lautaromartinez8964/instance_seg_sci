我先把当前 augment 配置和数据管线再对齐一遍，然后按“每个 epoch 实际喂了什么、增强在哪一步发生、总图片数会不会变”这三个问题拆开讲清楚。

读取 [](file:///home/yxy18034962/projects/mmdetection/configs/_base_/datasets/gravel_instance_414_augment.py)

读取 [](file:///home/yxy18034962/projects/mmdetection/configs/gravel/mask_rcnn_r50_fpn_36e_gravel_414_augment.py)

读取 [](file:///home/yxy18034962/projects/mmdetection/configs/_base_/datasets/gravel_instance_414.py)

我再补一个关键事实量化：确认训练集图片数和每个 epoch 的 iteration 数，这样能把“36 个 epoch 到底喂了多少次 batch”说完整。

Ran terminal command:  cd /home/yxy18034962/projects/mmdetection && source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmdec && python - <<'PY'
import json, math
from pathlib import Path
ann = Path('data/gravel_roboflow_414_mmdet/annotations/instances_train.json')
data = json.loads(ann.read_text())
images = len(data['images'])
print('train_images', images)
print('iters_per_epoch_batch2', math.ceil(images / 2))
print('samples_seen_over_36_epochs', images * 36)
print('optimizer_steps_over_36_epochs', math.ceil(images / 2) * 36)
PY

量化也确认了：train split 是 692 张图，batch=2 时每个 epoch 是 346 个 iteration。下面我直接按训练链路解释，不再做额外操作。增强主逻辑在 gravel_instance_414_augment.py，训练节奏在 mask_rcnn_r50_fpn_36e_gravel_414_augment.py。你现在这套训练可以理解成：数据集本身没有扩容成更多图片文件，而是每次取样时，在线随机生成一个增强版本，再喂给模型。

**先说最核心的结论**

1. 训练集原始图片总数还是 692 张，没有变。
2. batch size 是 2，所以每个 iteration 喂 2 张图。
3. 692 正好能被 2 整除，所以每个 epoch 是 346 个 iteration，没有尾巴 batch。
4. 36 个 epoch 的意思是，把这 692 张原始图作为“底图”完整遍历 36 次。
5. 所以从“原始样本被抽到多少次”这个角度看，总共会看到 692 × 36 = 24912 张样本。
6. 但这 24912 次里，大多数并不是同一张图的完全相同像素版本，而是每次在线随机增强后的版本。
7. 总优化步数是 346 × 36 = 12456 步。

**每个 epoch 到底怎么喂**

每个 epoch 开始时，训练 dataloader 会先把 692 张训练图打乱顺序。这里用的是 shuffle=True，同时还用了 AspectRatioBatchSampler，所以它不是简单地完全乱配，而是会尽量把宽高比接近的图放到一个 batch 里，减少后续 padding 浪费。

然后训练过程是这样走的：

1. 从 692 张图里按当前 epoch 的乱序顺序取 2 张图。
2. 对这 2 张图分别独立执行整条增强流水线。
3. 增强完成后，把这 2 张图组成一个 batch。
4. 模型前向、反向、更新一次参数，这就是 1 个 iteration。
5. 做满 346 个 iteration，这个 epoch 结束。
6. 接着做一次验证，因为你的配置里 val_interval 是 1，也就是每个 epoch 都验证一次。

所以“每个 epoch 输入图片总数变化吗”这个问题，答案是：

不变化。每个 epoch 作为底图被抽到的还是 692 张，只是这 692 张在进入模型前会被随机变形成 692 个增强版本。

**增强链路是按什么顺序发生的**

当前训练流水线顺序是这样的：

1. LoadImageFromFile
2. LoadAnnotations
3. RandomChoiceResize
4. Albu
5. RandomFlip
6. PhotoMetricDistortion
7. PackDetInputs

也就是说，一张图真正进模型前，先读图和标注，再做多尺度缩放，再做 Albumentations 增强块，再做 MMDetection 自己的水平翻转，再做光照颜色扰动，最后打包成模型输入。

**把每一步拆开讲**

1. 读图和标注

先读原图，再把 bbox 和 mask 标注一起读进来。因为你是实例分割，增强时不是只改图，还要同步改框和掩码。

2. 多尺度缩放 RandomChoiceResize

当前会从这 5 个尺度里随机选 1 个：

480
576
640
704
800

注意这里不是把图片粗暴拉成固定正方形，而是 keep_ratio=True，保持原始宽高比。更准确地说，是把图缩放到“落在该目标尺度约束下”的大小，所以同一张图这次可能按 480 档输入，下次可能按 800 档输入。

这一步的作用是让模型在不同目标尺寸下都见过同一类碎石目标，提升尺度鲁棒性。

3. Albu 几何增强块

这一块的外层是 OneOf，触发概率是 0.7。意思不是三个都做，而是：

有 70% 的概率进入这个块；
一旦进入，就在下面三种里随机选一种做。

三种几何增强是：

VerticalFlip
也就是上下翻转。

RandomRotate90
也就是按 90 度整数倍旋转。

ShiftScaleRotate
轻量平移、缩放、旋转。
其中参数是：
平移幅度上限 6.25%
缩放幅度上限 10%
旋转角度上限 15 度

因为这三项子变换的权重一样，所以粗略理解就是：
每张图大约有 23.3% 的概率做上下翻转，
23.3% 的概率做 90 度旋转，
23.3% 的概率做轻量仿射，
30% 的概率这一块完全不做。

这部分是你这次“几何增强”的核心。

4. Albu 外观多样性块

这一块也是 OneOf，触发概率是 0.6。进入后，从 4 个外观增强里随机挑 1 个：

RandomBrightnessContrast
调亮度和对比度。

HueSaturationValue
调色相、饱和度、明度。

RGBShift
分别对 R、G、B 三个通道做轻量偏移。

CLAHE
局部对比度增强。

所以这一步不是每次把四个都堆上去，而是：
60% 的概率做其中一个，
40% 的概率这一块不做。

平均下来每个子增强大概是 15% 左右的命中率。

这部分主要是在模拟不同拍摄条件、材质反光、曝光和颜色漂移。

5. Albu 轻退化块

这一块触发概率是 0.25，进入后在 4 个里选 1 个：

JPEG 压缩退化
MotionBlur
MedianBlur
Blur

也就是说，这一块是轻度使用，不是重手退化。平均每个子项大概 6.25% 左右命中率。

它的目的不是把图搞坏，而是让模型对轻微模糊、压缩损失、成像退化更稳。

6. MMDetection 自己的 RandomFlip

在 Albu 之后，还有一层 RandomFlip，概率 0.5。

这层默认就是水平翻转。也就是说，即使前面的几何块没触发，这里仍然有 50% 的概率做一次左右翻转。

所以当前几何增强并不是只有一种翻转，而是：

Albu 里的垂直翻转
Albu 里的 90 度旋转
Albu 里的轻量仿射
MMDet 里的水平翻转

这几类一起构成你的几何增强。

7. PhotoMetricDistortion

这一步是 MMDetection 自带的颜色扰动，它内部不是“必然做满所有子操作”，而是包含随机亮度、对比度、饱和度、色调等扰动逻辑。

所以你现在其实有两层颜色外观增强：

一层是 Albu 的外观 OneOf
一层是 MMDet 的 PhotoMetricDistortion

这就是为什么我说你现在的增强不是单一加法，而是“几何 + 外观多样性 + 轻退化”的组合链路。

**输入图片总数为什么没变，但模型看到的东西又变多了**

这是很多人一开始最容易混淆的点。

数据集文件层面：
还是 692 张 train 图，没有多出任何新图片文件。

训练看到的样本层面：
同一张底图每次被取到时，都会重新随机决定：
选哪个 resize 尺度；
几何块做不做，做哪一种；
外观块做不做，做哪一种；
退化块做不做，做哪一种；
水平翻转做不做；
PhotoMetricDistortion 内部怎么扰动。

所以同一张原图在第 1 个 epoch 和第 17 个 epoch 被喂进去，通常不是同一个像素版本。

这就是在线增强的本质：
图片文件数不变，但样本分布被不断随机展开。

**每个 epoch 会不会“越来越多图”**

不会。

每个 epoch 还是那 692 张底图各走一遍。
不会变成 692 乘上若干增强倍数。
不会把每张图复制出 5 个、10 个固定增强副本再训练。
也没有用 Mosaic、MixUp、Copy-Paste 这种会把多张图合成一张的新样本机制。

所以你这次不是“扩容数据集”，而是“让每次看到的数据版本更随机”。

**从一张图进入模型的完整链路，你可以这样记**

一张 gravel 训练图进入模型时，链路是：

原图与标注
先随机选一个尺度缩放
再有 70% 概率做一种几何增强
再有 60% 概率做一种颜色外观增强
再有 25% 概率做一种轻退化
再有 50% 概率做一次水平翻转
再做一层 PhotoMetricDistortion
最后打包进 Mask R-CNN

所以“当前增强逻辑”不是某一个大增强，而是多层随机门控串起来的。

**和你之前的 baseline 相比，增强具体多了什么**

baseline 在 gravel_instance_414.py 里，只有：

3 档多尺度缩放：512、640、768
水平翻转 0.5
PhotoMetricDistortion

augment 版相比它，新增了三件事：

1. 更宽的尺度覆盖
从 3 档扩大到 5 档，而且最低到 480，最高到 800。

2. 更强的几何变化
新增垂直翻转、90 度旋转、轻量平移缩放旋转。

3. 更丰富的成像变化
新增亮度对比度、HSV、RGBShift、CLAHE，以及 JPEG 压缩和轻模糊。

**一个容易忽略但重要的细节**

你的 Albu 配置里用了两项保护：

filter_lost_elements=True
skip_img_without_anno=True

意思是，如果某次几何变换把目标框或 mask 搞到不可用了，会过滤掉这些失效标注；如果增强后整张图已经没有有效标注了，会避免把这种无意义样本直接送进训练。

这能减少因为旋转、平移导致的坏样本。

**最后用一句话概括**

你现在这套 36 epoch、batch=2 的训练，本质上是：
用 692 张原始训练图，在每个 epoch 打乱后分成 346 个 batch；每张图在被取出的那个瞬间，在线随机经历“多尺度 + 几何 + 外观 + 轻退化 + 水平翻转 + 光照颜色扰动”这条链路，再送进 Mask R-CNN。图片文件数没变，但模型每轮看到的输入分布比 baseline 丰富得多。

如果你愿意，我下一步可以直接把这段解释整理写进你当前打开的 gravel_roboflow数据增强.md，顺手补一张“训练输入链路图”。Continued with Autopilot我刚才已经把增强链路、每个 epoch 的输入方式、图片总数是否变化、以及 36 epoch 与 batch=2 的实际含义讲清楚了。现在补上完成标记。



我刚才已经把当前增强版训练的链路解释完整了，包括：

1. 当前增强是怎么串起来的。
2. 36 个 epoch、batch=2 时每个 epoch 怎么喂数据。
3. 为什么每个 epoch 还是 692 张底图，但模型实际看到的输入版本会不断变化。
4. 每个 epoch 的 iteration 数、36 epoch 的总样本数和总优化步数。

如果你要继续往下走，我建议直接做其中一个：
1. 我把这段整理写进 gravel_roboflow数据增强.md。
2. 我再给你画成一版“训练输入链路图 + 概率树”，更直观。