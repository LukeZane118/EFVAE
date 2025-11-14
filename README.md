## [EFVAE: Efficient Federated Variational Autoencoders For Collaborative Filtering [PDF]](https://dl.acm.org/doi/10.1145/3627673.3679818)

## 1. Overview
This repository is a PyTorch Implementation for "[EFVAE: Efficient Federated Variational Autoencoders For Collaborative Filtering (CIKM2024)](https://dl.acm.org/doi/10.1145/3627673.3679818)".

**Authors**: Lu Zhang, Qian Rong, Xuanang Ding, Guohui Li, and Ling Yuan \
**Codes**: https://github.com/LukeZane118/EFVAE

Note: this project is built upon [FMSS](https://github.com/LachlanLin/FMSS), [rectorch](https://github.com/makgyver/rectorch), and [RecBole](https://github.com/RUCAIBox/RecBole).

## 2. Environment:

The code was developed and tested on the following python environment: 
```
python 3.8.13
pytorch 1.8.1
colorlog 6.6.0
colorama 0.4.5
pandas 1.2.3
numpy 1.21.5
scipy 1.9.0
munch 2.5.0
Bottleneck 1.3.4
scikit_learn 0.23.2
numba 0.55.2
fast_pytorch_kmeans 0.2.0.1
```

## 3. Instructions:

Train and evaluate EFVAE and other baselines:
```
bash ./run.sh
```

## 4. Citation

If you find this code useful in your research, please cite the following paper:
```
@inproceedings{zhang2024efvae,
  title={EFVAE: Efficient Federated Variational Autoencoder for Collaborative Filtering},
  author={Zhang, Lu and Rong, Qian and Ding, Xuanang and Li, Guohui and Yuan, Ling},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={3176--3185},
  year={2024}
}
```
