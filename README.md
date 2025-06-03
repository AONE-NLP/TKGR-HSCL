# HSCL

The code of HSCL
Improving Temporal Knowledge Graph Reasoning with Hierarchical Semantic-Aware Contrastive Learning

## Commands

```
nohup python src/main.py -d {dataset} --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --topk 10 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --batch-size 1 --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --angle 10 --discount 1 --add-static-graph --temperature 0.07 --lambda-cl 0.5 --lambda-pcl 0.5 --test > output.log 2>&1 & 
```

## Acknowledge
The basic framework of our code is referenced from HGLS https://github.com/CRIPAC-DIG/HGLS.

## Citation


