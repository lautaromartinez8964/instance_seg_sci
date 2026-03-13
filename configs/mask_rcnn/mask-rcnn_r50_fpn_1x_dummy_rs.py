_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 1.修改模型类别数为2
model = dict(
    roi_head = dict(
        bbox_head = dict(num_classes=2),
        mask_head=dict(num_classes=2)
    )
)

# 2.数据集类型和类别
metainfo = dict(classes=('ship', 'tank'), palette=[(200,50,50), (150,150,150)])
data_root = 'data/dummy_rs/'

# 修改Dataloader
train_dataloader = dict(
    batch_size=4,
    dataset = dict(
        data_root = data_root,
        metainfo = metainfo,
        ann_file = 'annotations/train.json',
        data_prefix = dict(img='train/')
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/train.json')
test_evaluator = val_evaluator

# 3. 训练策略：只跑 5 个 Epoch（为了快）
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=5)
optim_wrapper = dict(optimizer=dict(lr=0.005))

# 4. 🚀 导师的秘密武器：迁移学习 (Transfer Learning)
# 加载我们刚才下载的 COCO 预训练权重作为初始化！
load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'