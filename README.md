
# Meta Continual Learning Revisited: Implicitly Enhancing Online Hessian Approximation via Variance Reduction 
ICLR'24 (Oral): Meta Continual Learning Revisited: Implicitly Enhancing Online Hessian Approximation via Variance Reduction  (Official Pytorch implementation).  



If you find this code useful in your research then please cite  
```bash
@inproceedings{wu2023meta,
  title={Meta Continual Learning Revisited: Implicitly Enhancing Online Hessian Approximation via Variance Reduction},
  author={Wu, Yichen and Huang, Long-Kai and Wang, Renzhen and Meng, Deyu and Wei, Ying},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
``` 


## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch/ Torchvision


## Running our method on benchmark datasets (CIFAR-10/100 & TinyImageNet).
Here is an example:
```bash
cd utils
python3 main.py --model vrmcl --dataset seq-cifar100 --n_epochs 1 --grad_clip_norm 1 --buffer_size 1000 --batch_size 32 --replay_batch_size 32 --lr 0.25 --alpha_init 0.1 --seed 0 --asyn_update --second_order --meta_update_per_batch 1 --inner_batch_size 8 --s_momentum 0.15 --s_lr 0.35
python3 main.py --model vrmcl --dataset seq-tinyimg --n_epochs 1 --grad_clip_norm 1 --buffer_size 1000 --batch_size 32 --replay_batch_size 32 --lr 0.25 --alpha_init 0.1 --seed 0 --asyn_update --second_order --meta_update_per_batch 1 --inner_batch_size 8 --s_momentum 0.15 --s_lr 0.35
python3 main.py --model vrmcl --dataset seq-cifar10 --n_epochs 1 --grad_clip_norm 1 --buffer_size 1000 --batch_size 32 --replay_batch_size 32 --lr 0.25 --alpha_init 0.1 --seed 0 --asyn_update --second_order --meta_update_per_batch 1 --inner_batch_size 8 --s_momentum 0.15 --s_lr 0.35
```

The default network structure is Reduced-ResNet18


## Acknowledgements
We thank the Pytorch Continual Learning framework *Mammoth*(https://github.com/aimagelab/mammoth)

