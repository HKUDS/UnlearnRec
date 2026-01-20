cd ../..
python main_drop.py\
    --trained_model ./checkpoints/gowalla/before_unlearning/pretrain_gowalla_lightgcn_advlightgcn0.5_reg1e-7_lr1e-3_b4096_ep200_dim128_ly3\
    --save_path ./checkpoints/gowalla/pretrain_4_unlearning/tmp1\
    --seed 1234\
    --adversarial_attack True\
    --fineTune False\
    --adv_method lightgcn0.5\
    --model lightgcn \
    --data gowalla \
    --reg 0.0000001 \
    --lr 0.001 \
    --batch 1024 \
    --epoch 256 \
    --sim_epoch 5\
    --latdim 128 \
    --gnn_layer 3\
    --unlearn_layer 0\
    --bpr_wei 1 \
    --align_type v2\
    --unlearn_type v1 \
    --unlearn_wei 1\
    --align_wei 0.005\
    --align_temp 1\
    --hyper_temp 1\
    --unlearn_ssl 0.001\
    --pretrain_drop_rate 0.2\
    --layer_mlp 2\
    --perf_degrade 0.5\
    --overall_withdraw_rate 0.1\
    --withdraw_rate_init 1\
    --leaky 0.99 2>&1 | tee ./logs/tmp1.log

cd -