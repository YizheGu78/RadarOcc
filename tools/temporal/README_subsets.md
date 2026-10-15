# DG-STF: 20% temporal PKL subsets

Generate fixed, deterministic small-data annotations for **train** and **validation** only. Leave the official test set untouched. The script has no extra dependencies beyond the Python standard library.

Run from the RadarOcc repository root:

~~~bash
conda activate radarocc5060

# Preview real scene/frame counts. Does not write PKLs.
python tools/temporal/create_temporal_subsets.py --dry-run

# Write two subset PKLs and their scene-selection manifests.
python tools/temporal/create_temporal_subsets.py --seed 42
~~~

Default sources:
- data/annotations/kradar_dict_train_official_temporal_doppler8.pkl
- data/annotations/kradar_dict_val_official_temporal_doppler8.pkl

Default generated files:
- data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl
- data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl

Each generated PKL has a sibling .selection.json manifest listing selected scenes, selected and total frame counts, random seed, and the **actual achieved fraction**. The script does not overwrite existing outputs unless --overwrite is passed. Custom source/destination paths: --train-pkl, --val-pkl, --output-dir.

## Train P1 on the 20% subset

The current temporal config can use the subsets without modifying the original config:

~~~bash
bash tools/dist_train.sh \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py \
  1 \
  --work-dir work_dirs/radarocc_small_t4_p1_screen20 \
  --cfg-options \
  data.train.ann_file=data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl \
  data.val.ann_file=data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl
~~~

Use the **same 20% validation subset** for all P1–P5 architecture comparisons. Do not use the official test IoU to select the architecture; after selection, retrain the winner on 100% train, select the checkpoint using validation, and run the official 100% test for the final RadarOcc comparison.

## Why complete scenes?

TemporalKRadarDataset generates T=4 frame windows by sorting scene_id/scene_token and order_in_scene. Random individual-frame subsampling can silently treat widely separated time steps as consecutive, corrupting history matching.

This tool selects **complete scenes**, retains original info ordering, and does not alter poses, radar/GT paths, Doppler, metadata or other record fields. Depending on scene sizes, exact 20% may be impossible: **review the actual percent and selected scene coverage**. It warns if the source itself contains gaps in order_in_scene; it does not introduce new gaps. Whole-scene sampling also does not guarantee balance of foreground/background or driving conditions.

## CPU-only tests

~~~bash
python -m unittest discover -s tests -p 'test_temporal_subset_pkls.py'
~~~
