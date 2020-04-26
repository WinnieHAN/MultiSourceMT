TOTAL_NUM_UPDATES=4000000
#20000 
WARMUP_UPDATES=500
LR=3e-05
#3e-05
MAX_TOKENS=2048
#2048
UPDATE_FREQ=4
BART_PATH=/home/projects/11001764/wenjuan/gec_wj/gec_combine/bart_combine/bart.large/model.pt
#/home/projects/11001764/gec_wj/gec_combine/bart_combine/checkpoints_freeze_qsub_lr306/checkpoint_last.pt
#/home/projects/11001764/gec_wj/gec_combine/bart_combine/bart.large/model.pt
#CUDA_VISIBLE_DEVICES=0,1,2,
/home/users/nus/dcshanw/miniconda3/envs/gec/bin/python /home/projects/11001764/wenjuan/gec_wj/gec_combine/bart_combine/train_freeze_parallel.py /home/projects/11001764/wenjuan/gec_wj/fine_tune_data/pretrain-bin \
	    --arch bart_large \
	    --restore-file $BART_PATH \
	    --max-tokens $MAX_TOKENS \
	    --task translation \
	    --source-lang src --target-lang trg \
	    --truncate-source \
	    --layernorm-embedding \
	    --share-all-embeddings \
	    --share-decoder-input-output-embed \
	    --reset-optimizer --reset-dataloader --reset-meters \
	    --required-batch-size-multiple 1 \
	    --criterion label_smoothed_cross_entropy \
	    --label-smoothing 0.1 \
	    --dropout 0.1 --attention-dropout 0.1 \
	    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
	    --clip-norm 0.1 \
	    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
	    --fp16 --update-freq $UPDATE_FREQ \
	    --skip-invalid-size-inputs-valid-test \
	    --find-unused-parameters \
	    --save-dir /home/projects/11001764/wenjuan/gec_wj/gec_combine/bart_combine/pretrain
