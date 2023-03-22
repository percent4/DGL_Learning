### 图深度学习框架

- PyG(PyTorch Geometric)
- DGL(Deep Graph Learning)
- NeuGraph
- EnGN
- AliGraph
- PGL (paddle graph learning)
- Galileo

DGL是纽约大学和亚马逊公司等共同开发的图深度学习框架，先后集成了PyTorch, MXNet, TensorFlow三个主流的深度学习框架，使用时用户可以灵活选择自己熟悉的框架作为DGL的训练后端。其特点如下：

- 灵活的图表示：DGL可以处理各种类型的图数据，包括有向图、无向图和多重图。同时，它支持不同类型的节点和边上的特征数据。
- 易于使用的API：DGL提供了简单易用的API，使用户能够方便地创建图和图神经网络，并进行训练和评估。
- 高效的计算：DGL使用高效的图计算引擎，使得对大规模图进行训练和推理成为可能。它还支持GPU加速，以加快计算速度。
- 可扩展性：DGL可以与其他深度学习框架（如PyTorch和TensorFlow）进行无缝集成，从而使得用户能够轻松地将图神经网络与其他深度学习模型结合起来。
- 社区支持：DGL有一个活跃的社区，包括开发人员和用户，他们共同致力于推动图神经网络的研究和应用。


### DGL入门

#### DGL安装

```
pip3 install dgl
```

需要注意的是，dgl要求torch的版本在1.9.0及以上。

参考网址：[https://docs.dgl.ai/install/index.html#](https://docs.dgl.ai/install/index.html#)

#### 图

##### 图的基本概念

DGL使用一个唯一的整数来表示一个节点，称为点ID；并用对应的两个端点ID表示一条边。同时，DGL也会根据边被添加的顺序，给每条边分配一个唯一的整数编号，称为边ID。节点和边的ID都是从0开始构建的。在DGL的图里，所有的边都是有方向的，即边(u,v)表示它是从节点u指向节点v的。

对于多个节点，DGL使用一个一维的整型张量来保存图的点ID，DGL称之为”节点张量”。为了指代多条边，DGL使用一个包含2个节点张量的元组 (U,V)，其中，用(U[i],V[i])指代一条 U[i]到 V[i]的边。

![https://data.dgl.ai/asset/image/user_guide_graphch_1.png](https://data.dgl.ai/asset/image/user_guide_graphch_1.png)

```python
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch as th

# 边 0->1, 0->2, 0->3, 1->3
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))
print(g) # 图中节点的数量是DGL通过给定的图的边列表中最大的点ID推断所得出的

# 获取节点的ID
print(g.nodes())

# 获取边的对应端点
print(g.edges())

# 获取边的对应端点和边ID
print(g.edges(form='all'))

# 如果具有最大ID的节点没有边，在创建图的时候，用户需要明确地指明节点的数量。
g = dgl.graph((u, v), num_nodes=8)
print(g)
```
输出结果：
```
Graph(num_nodes=4, num_edges=4,
      ndata_schemes={}
      edata_schemes={})
tensor([0, 1, 2, 3])
(tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]))
(tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]), tensor([0, 1, 2, 3]))
Graph(num_nodes=8, num_edges=4,
      ndata_schemes={}
      edata_schemes={})
```

##### 有向图转换为无向图

对于无向的图，用户需要为每条边都创建两个方向的边。可以使用 dgl.to_bidirected() 函数来实现这个目的。 

```python
bg = dgl.to_bidirected(g)
bg.edges()
```

##### 数据类型转换

DGL支持使用32位或64位的整数作为节点ID和边ID。节点和边ID的数据类型必须一致。如果使用64位整数， DGL可以处理最多 2^63 − 1个节点或边。不过，如果图里的节点或者边的数量小于2^31 − 1，用户最好使用32位整数。 这样不仅能提升速度，还能减少内存的使用。

```python
# 数据类型转换
edges = th.tensor([2, 5, 3]), th.tensor([3, 5, 0])  # 边：2->3, 5->5, 3->0
g64 = dgl.graph(edges)  # DGL默认使用int64
print(g64.idtype)

g32 = dgl.graph(edges, idtype=th.int32)  # 使用int32构建图
g32.idtype

g64_2 = g32.long()  # 转换成int64
g64_2.idtype

g32_2 = g64.int()  # 转换成int32
g32_2.idtype
```
##### 节点和边的特征

DGLGraph对象的节点和边可具有多个用户定义的、可命名的特征，以储存图的节点和边的属性。 通过ndata和edata接口可访问这些特征。

```python
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
```
输出结果：
```
Graph(num_nodes=6, num_edges=4,
      ndata_schemes={}
      edata_schemes={})
Graph(num_nodes=6, num_edges=4,
      ndata_schemes={'x': Scheme(shape=(3,), dtype=torch.float32)}
      edata_schemes={'x': Scheme(shape=(), dtype=torch.int32)})
tensor([1., 1., 1.])
tensor([1, 1], dtype=torch.int32)
tensor([-1.9780,  1.3512,  0.1171, -0.4730, -0.5262])
```

关于ndata和edata接口的重要说明：

- 仅允许使用数值类型（如单精度浮点型、双精度浮点型和整型）的特征。这些特征可以是标量、向量或多维张量。

- 每个节点特征具有唯一名称，每个边特征也具有唯一名称。节点和边的特征可以具有相同的名称（如上述示例代码中的 'x' ）。

- 通过张量分配创建特征时，DGL会将特征赋给图中的每个节点和每条边。该张量的第一维必须与图中节点或边的数量一致。不能将特征赋给图中节点或边的子集。

- 相同名称的特征必须具有相同的维度和数据类型。

- 特征张量使用“行优先”的原则，即每个行切片储存1个节点或1条边的特征

##### 加权图的实现

```python
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch as th

# 边 0->1, 0->2, 0->3, 1->3
edges = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
weights = th.tensor([0.1, 0.6, 0.9, 0.7])  # 每条边的权重
g = dgl.graph(edges)
g.edata['w'] = weights  # 将其命名为 'w'
print(g)
print(g.edata['w'])
```
输出结果：
```
Graph(num_nodes=4, num_edges=4,
      ndata_schemes={}
      edata_schemes={'w': Scheme(shape=(), dtype=torch.float32)})
tensor([0.1000, 0.6000, 0.9000, 0.7000])
```

##### 从外部源创建图

可以从外部来源构造一个 DGLGraph 对象，包括：

- 从用于图和稀疏矩阵的外部Python库（NetworkX 和 SciPy）创建而来
- 从磁盘加载图数据

从SciPy稀疏矩阵和NetworkX图创建DGL图示例：

```python
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import scipy.sparse as sp
import networkx as nx

spmat = sp.rand(100, 100, density=0.05)     # 5%非零项
dg = dgl.from_scipy(spmat)                  # 来自SciPy
print(dg)
```
输出结果为：

```
Graph(num_nodes=100, num_edges=500,
      ndata_schemes={}
      edata_schemes={})
Graph(num_nodes=5, num_edges=8,
      ndata_schemes={}
      edata_schemes={})
```
`nx.path_graph(5)`构建了一个无向的NetworkX图 networkx.Graph ，而 DGLGraph 的边总是有向的。如需构建有向图，代码如下：

```python
nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])
dgl.from_networkx(nxg)
```

从磁盘加载图：

- CSV
- JSON/GML格式
- DGL二进制格式

#### 异构图

![https://data.dgl.ai/asset/image/user_guide_graphch_2.png](https://data.dgl.ai/asset/image/user_guide_graphch_2.png)

在DGL中，一个异构图由一系列子图构成，一个子图对应一种关系。每个关系由一个字符串三元组定义 (源节点类型, 边类型, 目标节点类型) 。由于这里的关系定义消除了边类型的歧义，DGL称它们为规范边类型。

```python
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch as th

# 创建一个具有3种节点类型和3种边类型的异构图
graph_data = {
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
}
g = dgl.heterograph(graph_data)
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)
print(g)
print(g.metagraph().edges())
```
输出结果为：
```
['disease', 'drug', 'gene']
['interacts', 'interacts', 'treats']
[('drug', 'interacts', 'drug'), ('drug', 'interacts', 'gene'), ('drug', 'treats', 'disease')]
Graph(num_nodes={'disease': 3, 'drug': 3, 'gene': 4},
      num_edges={('drug', 'interacts', 'drug'): 2, ('drug', 'interacts', 'gene'): 2, ('drug', 'treats', 'disease'): 1},
      metagraph=[('drug', 'drug', 'interacts'), ('drug', 'gene', 'interacts'), ('drug', 'disease', 'treats')])
[('drug', 'drug'), ('drug', 'gene'), ('drug', 'disease')]
```

异构图为管理不同类型的节点和边及其相关特征提供了一个清晰的接口。这在以下情况下尤其有用:

- 不同类型的节点和边的特征具有不同的数据类型或大小。
- 用户希望对不同类型的节点和边应用不同的操作。

如果上述情况不适用，并且用户不希望在建模中区分节点和边的类型，则DGL允许使用 dgl.DGLGraph.to_homogeneous() API将异构图转换为同构图。 具体行为如下:

- 用从0开始的连续整数重新标记所有类型的节点和边。
- 对所有的节点和边合并用户指定的特征。

### 在GPU上使用DGLGraph

任何涉及GPU图的操作都是在GPU上运行的。因此，这要求所有张量参数都已经放在GPU上，其结果(图或张量)也将在GPU上。此外，GPU图只接受GPU上的特征数据。

```python
import dgl
import torch as th
u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata['x'] = th.randn(5, 3)   # 原始特征在CPU上
g.device
cuda_g = g.to('cuda:0')         # 接受来自后端框架的任何设备对象
cuda_g.device
cuda_g.ndata['x'].device        # 特征数据也拷贝到了GPU上
# 由GPU张量构造的图也在GPU上
u, v = u.to('cuda:0'), v.to('cuda:0')
g = dgl.graph((u, v))
g.device
```
输出结果：
```
device(type='cpu')
device(type='cuda', index=0)
device(type='cuda', index=0)
device(type='cuda', index=0)
```

### 参考文献

1. 中文版用户使用手册：[https://docs.dgl.ai/guide_cn/graph.html](https://docs.dgl.ai/guide_cn/graph.html)