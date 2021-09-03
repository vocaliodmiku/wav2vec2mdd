# wav2vec2mdd
End-to-End Mispronunciation Detection via [wav2vec2.0](https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/wav2vec/README.md)

We provide some useful script for fine-tuning wav2vec2.0 on L2-ARCTIC.(process data/finetune/evaluate)
evaluate part are come from https://github.com/cageyoko/CTC-Attention-Mispronunciation
## Install Requirements

* [fairseq](https://github.com/pytorch/fairseq/blob/master/README.md)
* [Flashlight Python Bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python)
* Evaluating the trained model requires tool [kaldi](https://kaldi-asr.org)

## Fine-tune a pre-trained model with CTC
We provide some useful script for fine-tuning wav2vec2.0 on L2-ARCTIC.
<div style='display: none'>

### Prepare training data manifest
```
$ python l2_labels.py /path/to/waves --dest /manifest/path 
```
### Fine-tune a pre-trained model
Edit the run.sh
```bash
#!/usr/python/bin/

export CUDA_VISIBLE_DEVICES=1 # GPU device ID
DATASET=/manifest/path

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
and 
```bash
$ sh run.sh
```
### Evaluating a CTC model
Edit the evaluate.sh
```bash
#!/usr/python/bin/

# Evaluating the CTC model
export CUDA_VISIBLE_DEVICES=0
DATASET=/manifest/path
FAIRSEQ_PATH=/path/to/fairseq

python3 $FAIRSEQ_PATH/examples/speech_recognition/infer.py $DATASET --task audio_pretraining \
--nbest 1 --path /path/to/checkpoints/checkpoint_best.pt --gen-subset test --results-path $DATASET --w2l-decoder viterbi \
--lm-weight 0 --word-score -1 --sil-weight 0 --criterion ctc --labels phn --max-tokens 640000

# Env 
export KALDI_ROOT=/path/to/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

# calculate the result of MDD
python3 result.py
align-text ark:ref.txt  ark:annotation.txt ark,t:- | wer_per_utt_details.pl > ref_human_detail
align-text ark:annotation.txt  ark:hypo.txt ark,t:- | wer_per_utt_details.pl > human_our_detail
align-text ark:ref.txt  ark:hypo.txt ark,t:- | wer_per_utt_details.pl > ref_our_detail
python3 ins_del_sub_cor_analysis.py
rm ref_human_detail human_our_detail ref_our_detail
```
and 
```bash
$ sh evaluate.sh >> result
```

   
# What's more
we are going to make wav2vec2-based model to provide dignose information in near future, Please stay tuned.
    
