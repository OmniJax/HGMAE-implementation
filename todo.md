- [x] dgl跑m2v

- [x] 植入到HGMAE

- [ ] 尝试openhgnn HAN代替

- [ ] +norm

```python
def preprocess_features(features):
    rowsum=torch.sum(features,dim=1)
    r_inv=torch.pow(rowsum,-1).flatten()
    r_inv=torch.where(torch.isinf(r_inv),0.,r_inv)
    r_mat_inv=torch.diag(r_inv)
    features*=features
    return features


    即行向量归一化    
```

```python
def normalize_adj(adj):
    rowsum=torch.sum(adj,dim=1)
    d_inv_sqrt=torch.pow(rowsu,-0.5).flatten()
    d_inv_sqrt=torch.where(torch.isinf(d_inv_sqrt),0.,d_inv_sqrt)
```

- [x] 论文中的acm数据，你看看能不能把它处理成dglgraph形式。然后再用dgl读取元路径的方式读出元路径的邻接矩阵

- [x] 如果都不信的话，那dglgraph这个对象有一个edata成员，这里面可以存储一些元路径矩阵之类的信息。

！用openhgnn的sampler。排除了是dgl m2v模型的问题

None 变成 累加

石老师好，我是陈泽琦，申请暑期住宿，我的目标：尽快找到自己的科研方向 。任务：目前正在复现模型，尽快使模型的性基论性。平时也需要多读文献，尽快找到新的科研任务。
