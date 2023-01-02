# SDMP
This is the offical implementation of 	
["A Simple Data Mixing Prior for Improving Self-Supervised Learning"](https://cihangxie.github.io/data/SDMP.pdf)
by [Sucheng Ren](https://oliverrensu.github.io/), [Huiyu Wang](https://csrhddlam.github.io/), [Zhengqi Gao](https://zhengqigao.github.io/), [Shengfeng He](http://www.shengfenghe.com/), [Alan Yuille](http://www.cs.jhu.edu/~ayuille/), [Yuyin Zhou](https://yuyinzhou.github.io/), [Cihang Xie](https://cihangxie.github.io/)

![teaser](method.png)

## MoCo with SDMP
Installing packages as [Moco v3](https://arxiv.org/abs/2104.02057).

### Training ViT small on 4 nodes.
```
cd moco
# ImageNet  
# On the first node
python main_moco.py \
  -a vit_small \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://hostnode:port' \
  --multiprocessing-distributed --world-size 4 --rank 0 \
  /path/to/imagenet
# On the rest node --rank 1, 2, 3

# On single node
python main_moco.py \
  -a vit_small -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  /path/to/imagenet
```

### Testing
```
python main_lincls.py \
  -a vit_small --lr=3 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained /path/to/checkpoint.pth.tar \
  /path/to/imagenet
  ```

### Checkpoints

The checkpoint of SDMP with 100 epochs pretraining (--epochs=100) is at [Google Drive](https://drive.google.com/drive/folders/1JLbyz5XiJd5D6HyazABSTxc1z0vH-t6S?usp=sharing). We also release the checkpoint of moco with 100 epochs pretraining for comparison. We make 1.2\% improvements. The checkpoint of 300 epochs is coming soon.


If you have any question, feel free to email [Sucheng Ren](oliverrensu@gmail.com)

## Citing SDMP
If you think think this paper and responsity is useful, please cite 
```
@inproceedings{ren2022sdmp,
  title     = {A Simple Data Mixing Prior for Improving Self-Supervised Learning},
  author    = {Ren, Sucheng and Wang, Huiyu and Gao, Zhengqi and He, Shengfeng and Yuille, Alan and Zhou, Yuyin and Xie, Cihang},
  booktitle = {CVPR},
  year      = {2022}
}
```
