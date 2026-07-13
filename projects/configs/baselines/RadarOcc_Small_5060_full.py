point_cloud_range = [0, -25.6, -2.6, 51.2, 25.6, 3.0]
class_names = [
    'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle', 'Pedestrian',
    'Pedestrian Group', 'Bicycle Group', 'Unknow'
]
dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadSparseRadar', to_float32=True),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        use_semantic=True,
        occ_path='data/K-RadarOcc/train',
        grid_size=[128, 128, 14],
        use_vel=False,
        unoccupied=0,
        pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
        cal_visible=False),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=[
            'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
            'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
        ]),
    dict(type='Collect3D', keys=['gt_occ', 'sparse_radar'])
]
test_pipeline = [
    dict(type='LoadSparseRadar', to_float32=True),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        use_semantic=True,
        occ_path='data/K-RadarOcc/train',
        grid_size=[128, 128, 14],
        use_vel=False,
        unoccupied=0,
        pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
        cal_visible=False),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=[
            'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
            'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
        ],
        with_label=False),
    dict(
        type='Collect3D',
        keys=['gt_occ', 'sparse_radar'],
        meta_keys=[
            'pc_range', 'occ_size', 'scene_token', 'lidar_token', 'radar_path',
            'occ_path'
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
            'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=({
        'type':
        'NuscOCCDataset',
        'data_root':
        'data/nuscenes/',
        'occ_root':
        'data/K-RadarOcc/train',
        'ann_file':
        'data/annotations/kradar_dict_train_doppler8.pkl',
        'pipeline': [{
            'type': 'LoadSparseRadar',
            'to_float32': True
        }, {
            'type': 'LoadOccupancy',
            'to_float32': True,
            'use_semantic': True,
            'occ_path': 'data/K-RadarOcc/train',
            'grid_size': [128, 128, 14],
            'use_vel': False,
            'unoccupied': 0,
            'pc_range': [0, -25.6, -2.6, 51.2, 25.6, 3.0],
            'cal_visible': False
        }, {
            'type':
            'OccDefaultFormatBundle3D',
            'class_names': [
                'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
                'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
            ]
        }, {
            'type': 'Collect3D',
            'keys': ['gt_occ', 'sparse_radar']
        }],
        'classes': [
            'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
            'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
        ],
        'modality': {
            'use_lidar': False,
            'use_camera': False,
            'use_radar': False,
            'use_map': False,
            'use_external': False
        },
        'test_mode':
        False,
        'use_valid_flag':
        True,
        'occ_size': [128, 128, 14],
        'pc_range': [0, -25.6, -2.6, 51.2, 25.6, 3.0],
        'box_type_3d':
        'LiDAR'
    }, ),
    val=dict(
        type='NuscOCCDataset',
        ann_file='data/annotations/kradar_dict_val_doppler8.pkl',
        pipeline=[
            dict(type='LoadSparseRadar', to_float32=True),
            dict(
                type='LoadOccupancy',
                to_float32=True,
                use_semantic=True,
                occ_path='data/K-RadarOcc/train',
                grid_size=[128, 128, 14],
                use_vel=False,
                unoccupied=0,
                pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
                cal_visible=False),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
                    'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['gt_occ', 'sparse_radar'],
                meta_keys=[
                    'pc_range', 'occ_size', 'scene_token', 'lidar_token',
                    'radar_path', 'occ_path'
                ])
        ],
        classes=[
            'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
            'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        occ_root='data/K-RadarOcc/train',
        data_root='data/nuscenes/',
        occ_size=[128, 128, 14],
        pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0]),
    test=dict(
        type='NuscOCCDataset',
        data_root='data/nuscenes/',
        ann_file='data/annotations/kradar_dict_test_doppler8.pkl',
        pipeline=[
            dict(type='LoadSparseRadar', to_float32=True),
            dict(
                type='LoadOccupancy',
                to_float32=True,
                use_semantic=True,
                occ_path='data/K-RadarOcc/train',
                grid_size=[128, 128, 14],
                use_vel=False,
                unoccupied=0,
                pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
                cal_visible=False),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
                    'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['gt_occ', 'sparse_radar'],
                meta_keys=[
                    'pc_range', 'occ_size', 'scene_token', 'lidar_token',
                    'radar_path', 'occ_path'
                ])
        ],
        classes=[
            'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
            'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        occ_root='data/K-RadarOcc/train',
        occ_size=[128, 128, 14],
        pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0]),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadSparseRadar', to_float32=True),
        dict(
            type='LoadOccupancy',
            to_float32=True,
            use_semantic=True,
            occ_path='data/K-RadarOcc/train',
            grid_size=[128, 128, 14],
            use_vel=False,
            unoccupied=0,
            pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
            cal_visible=False),
        dict(
            type='OccDefaultFormatBundle3D',
            class_names=[
                'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
                'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['gt_occ', 'sparse_radar'],
            meta_keys=[
                'pc_range', 'occ_size', 'scene_token', 'lidar_token',
                'radar_path', 'occ_path'
            ])
    ],
    save_best='SSC_mean',
    rule='greater')
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/radarocc_small_5060_full'
load_from = None
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/occ_plugin/'
img_norm_cfg = None
occ_path = 'data/K-RadarOcc/train'
train_ann_file = 'data/annotations/kradar_dict_train_doppler8.pkl'
val_ann_file = 'data/annotations/kradar_dict_val_doppler8.pkl'
LIST_CLS_NAME = [
    'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle', 'Pedestrian',
    'Pedestrian Group', 'Bicycle Group', 'Unknow'
]
occ_size = [128, 128, 14]
voxel_channels = [80, 160, 320, 640]
empty_idx = 0
num_cls = 3
visible_mask = False
cascade_ratio = 1
sample_from_voxel = False
sample_from_img = False
numC_Trans = 144
_dim_ = 144
voxel_out_channel = 196
voxel_out_indices = (0, 1, 2, 3)
_pos_dim_ = 48
_ffn_dim_ = 288
_num_cams_ = 5
_num_layers_self_ = 1
_num_layers_cross_ = 2
_num_points_self_ = 8
model = dict(
    type='RadarOcc_small',
    loss_norm=True,
    embed_dims=144,
    pts_middle_encoder=dict(
        type='RadarEncV8small',
        input_channel=11,
        base_channel=32,
        out_channel=144,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[512, 512, 56]),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=144,
        block_inplanes=[80, 160, 320, 640],
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=[80, 160, 320, 640],
        out_channels=196,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=1,
        sample_from_voxel=False,
        sample_from_img=False,
        final_occ_size=[128, 128, 14],
        fine_topk=15000,
        empty_idx=0,
        num_level=4,
        in_channels=[196, 196, 196, 196],
        out_channel=3,
        point_cloud_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0),
        balance_cls_weight=True,
        class_num=3,
        class_names=[
            'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
            'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
        ],
        fp16=True),
    cross_transformer=dict(
        type='PerceptionTransformer3D',
        rotate_prev_bev=True,
        use_shift=True,
        embed_dims=144,
        num_cams=5,
        encoder=dict(
            type='VoxFormerEncoder3D',
            num_layers=1,
            pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
            num_points_in_pillar=10,
            return_intermediate=False,
            transformerlayers=dict(
                type='VoxFormerLayer3D',
                attn_cfgs=[
                    dict(
                        type='DeformCrossAttention3DCustom',
                        embed_dims=144,
                        num_levels=1,
                        num_points=4,
                        im2col_step=1,
                        num_heads=9)
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=144,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True)),
                feedforward_channels=288,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
    self_transformer=dict(
        type='PerceptionTransformer3D',
        rotate_prev_bev=True,
        use_shift=True,
        embed_dims=144,
        num_cams=5,
        encoder=dict(
            type='VoxFormerEncoder3D',
            num_layers=1,
            pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
            num_points_in_pillar=10,
            return_intermediate=False,
            transformerlayers=dict(
                type='VoxFormerLayer3D',
                attn_cfgs=[
                    dict(
                        type='DeformSelfAttention3DCustom',
                        fp16_enabled=False,
                        embed_dims=144,
                        num_levels=1,
                        num_points=4,
                        im2col_step=1,
                        num_heads=9)
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=144,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True)),
                feedforward_channels=288,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
    positional_encoding=dict(
        type='LearnedPositionalEncoding3D',
        num_feats=48,
        row_num_embed=128,
        col_num_embed=128,
        z_num_embed=20),
    empty_idx=0)
test_config = dict(
    type='NuscOCCDataset',
    occ_root='data/K-RadarOcc/train',
    data_root='data/nuscenes/',
    ann_file='data/annotations/kradar_dict_test_doppler8.pkl',
    pipeline=[
        dict(type='LoadSparseRadar', to_float32=True),
        dict(
            type='LoadOccupancy',
            to_float32=True,
            use_semantic=True,
            occ_path='data/K-RadarOcc/train',
            grid_size=[128, 128, 14],
            use_vel=False,
            unoccupied=0,
            pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0],
            cal_visible=False),
        dict(
            type='OccDefaultFormatBundle3D',
            class_names=[
                'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
                'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['gt_occ', 'sparse_radar'],
            meta_keys=[
                'pc_range', 'occ_size', 'scene_token', 'lidar_token',
                'radar_path', 'occ_path'
            ])
    ],
    classes=[
        'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
        'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=False,
        use_radar=False,
        use_map=False,
        use_external=False),
    occ_size=[128, 128, 14],
    pc_range=[0, -25.6, -2.6, 51.2, 25.6, 3.0])
train_config = ({
    'type':
    'NuscOCCDataset',
    'data_root':
    'data/nuscenes/',
    'occ_root':
    'data/K-RadarOcc/train',
    'ann_file':
    'data/annotations/kradar_dict_train_doppler8.pkl',
    'pipeline': [{
        'type': 'LoadSparseRadar',
        'to_float32': True
    }, {
        'type': 'LoadOccupancy',
        'to_float32': True,
        'use_semantic': True,
        'occ_path': 'data/K-RadarOcc/train',
        'grid_size': [128, 128, 14],
        'use_vel': False,
        'unoccupied': 0,
        'pc_range': [0, -25.6, -2.6, 51.2, 25.6, 3.0],
        'cal_visible': False
    }, {
        'type':
        'OccDefaultFormatBundle3D',
        'class_names': [
            'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
            'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
        ]
    }, {
        'type': 'Collect3D',
        'keys': ['gt_occ', 'sparse_radar']
    }],
    'classes': [
        'Static', 'Sedan', 'Bus or Truck', 'Motorcycle', 'Bicycle',
        'Pedestrian', 'Pedestrian Group', 'Bicycle Group', 'Unknow'
    ],
    'modality': {
        'use_lidar': False,
        'use_camera': False,
        'use_radar': False,
        'use_map': False,
        'use_external': False
    },
    'test_mode':
    False,
    'use_valid_flag':
    True,
    'occ_size': [128, 128, 14],
    'pc_range': [0, -25.6, -2.6, 51.2, 25.6, 3.0],
    'box_type_3d':
    'LiDAR'
}, )
optimizer = dict(
    type='AdamW',
    lr=0.0003,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=15)
custom_hooks = []
