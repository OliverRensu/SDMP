# SDMP
This is the offical implementation of 	
["A Simple Data Mixing Prior for Improving Self-Supervised Learning"](https://cihangxie.github.io/data/SDMP.pdf)
by Sucheng Ren</strong>, 
                  <a href="https://csrhddlam.github.io/">Huiyu Wang</a>, 
                  <a href="https://zhengqigao.github.io/">Zhengqi Gao</a>,
                  <a href="http://www.shengfenghe.com/">Shengfeng He</a>,
                  <a href="http://www.cs.jhu.edu/~ayuille/">Alan Yuille</a>,
                  <a href="https://yuyinzhou.github.io/">Yuyin Zhou</a>,
                  <a href="https://cihangxie.github.io/">Cihang Xie


## Plug-in example
Here we show the core part of our paper based on [Moco v3](https://arxiv.org/abs/2104.02057)

The extra version:
```
x1, x2, x3, x4 = data #from four different views  
# Naive loss
q1 = encoder(x1)
q1 = encoder(x2)
k1 = momentum_encoder(x1) #no_grad
k2 = momentum_encoder(x2) #no_grad
naive_loss = contrastive_loss(q1, k2)+contrastive_loss(q2, k1) # two views

# data mixing  mix x3 and x4 with the same lambda
lam, x3_mix, x4_mix = mix(x3, x3.flip(0), x4, x4.flip(0))
q3_mix = encoder(x3_mix)
k4 = momentum_encoder(x4) #no_grad
k4_mix = momentum_encoder(x4_mix) #no_grad

# Mix loss (Mix and Source)
ms_loss = contrastive_loss(q3, k4)*lam + contrastive_loss(q3, k4.flip(0)) *(1-lam)

# Mix loss (Intra-batch Mix)
 common = np.minimum(lam, 1-lam.clone().flip()) + np.minimum(1-lam, lam.clone().flip())
mb_loss = contrastive_loss(q3, k4_mix)*(1/(1+common)) + contrastive_loss(q3, k4_mix.flip(0)) *(common/(1+common))

m_loss = # one extra view
Loss = naive_loss + ms_loss + mb_loss
```


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
}
```
