# MetaBBo

### A project for experimenting and testing metaheuristics for Black Box optimization

### for now it consists of the following methods:

- RLEPSO

It consists of refactored and modified parts of the following repositories:
- <https://github.com/GMC-DRL/MetaBox>[[1]](#1)


### installation:

```bash
git clone https://github.com/GMC-DRL/MetaBox.git && cd MetaBBo && pip install .
```

### Example of an experiment

```bash
python3 metabbo/main.py --run_experiment --problem protein --difficulty easy --train_agent RLEPSO_Agent --train_optimizer RLEPSO_Optimizer
```




## References
<a id="1">[1]</a> 
Ma, Zeyuan and Guo, Hongshu and Chen, Jiacheng and Li, Zhenrui and Peng, Guojun and Gong, Yue-Jiao and Ma, Yining and Cao, Zhiguang (2023). 
MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning.
Advances in Neural Information Processing Systems, vol. 36.


