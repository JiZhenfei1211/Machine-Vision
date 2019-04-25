#!/usr/bin/env bash
export DATA_SET_ROOT='../A2D'
PYTHONPATH='./':$PYTHONPATH python train.py \
    --dataset A2D \
    --cfg model_cfgs/e2e_faster_rcnn_R-101-FPN_1x.yaml \
    --bs 8 \
    --nw 0 \
    --lr 3e-4 \
    --train_lst $DATA_SET_ROOT/list/train.txt \
    --annotation_root $DATA_SET_ROOT/Annotations/mat \
    --frame_root $DATA_SET_ROOT/pngs320H \
    --id_map_file $DATA_SET_ROOT/list/actor_id_action_id.txt \
    --dataset A2D \
    --use_tfboard \
    --segment_length 2 \
    --output_dir ./train_output \
    --load_detectron ./detectron_weights/model_final.pkl \
    --snapshot_iters 4750 
