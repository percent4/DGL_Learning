# -*- coding: utf-8 -*-
# @Time : 2023/3/22 15:11
# @Author : Jclian91
# @File : graph_test.py
# @Place : Gusu, Suzhou
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch as th
g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0])) # 6个节点，4条边
print(g)

g.ndata['x'] = th.ones(g.num_nodes(), 3)               # 长度为3的节点特征
g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32)  # 标量整型特征
print(g)

# 不同名称的特征可以具有不同形状
g.ndata['y'] = th.randn(g.num_nodes(), 5)
print(g.ndata['x'][1])
print(g.edata['x'][th.tensor([0, 3])])  # 获取边0和3的特征
print(g.ndata['y'][1])