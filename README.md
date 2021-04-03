# wav2vec2mdd
End-to-End Mispronunciation Detection via wav2vec2.0

## Install Requirements

* [fairseq](https://github.com/pytorch/fairseq/blob/master/README.md)
* [Flashlight Python Bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python)


## Fine-tune a pre-trained model with CTC
Edit the run.sh.
```bash
#!/usr/python/bin/

export CUDA_VISIBLE_DEVICES=1 # GPU device ID
DATASET=/manifest/path/

FAIRSEQ_PATH=/path/to/fairseq
valid_subset=valid
model_path=/path/to/pretrain_model.pt  # do not use finetuned model
config_dir=/path/to/config 

config_name=base_finetune # made by reffering https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/config/finetuning/base_10m.yaml
labels=phn
python3 $FAIRSEQ_PATH/fairseq_cli/hydra_train.py \
    distributed_training.distributed_port=0 \
    task.labels=$labels \
    task.data=$DATASET \
    dataset.valid_subset=$valid_subset \
    distributed_training.distributed_world_size=1 \
    model.w2v_path=$model_path \
    --config-dir $config_dir \
    --config-name $config_name
```
