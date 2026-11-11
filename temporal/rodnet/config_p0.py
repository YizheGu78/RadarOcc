_base_ = ['../p0_config.py']
model = dict(p0_temporal_cfg=dict(
    method='rodnet_tdc',
    bins_path='data/K-Radar-official-meta/resources/info_arr.mat',
    frame_nums=4, channels=32, kernel_size=3, chunk_size=1024,
    bev_bounds=(-4., -64., 84., 64.), cell_size=.8,
    height_bounds=(-4., 8.), height_cell_size=1.,
    checkpoint_chunks=True, max_log_correction=.5))
work_dir = 'work_dirs/rodnet_tdc_p0_height_20pct_scratch_seed42'
