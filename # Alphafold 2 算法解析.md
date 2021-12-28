# Alphafold 2 算法解析

---

> #### 写在前面
> 本文中所述的“Alphafold 2”指Deepmind团队开发的一个用于预测蛋白质结构的一个神经网络模型，该模型在CASP14中展现出了碾压性的优势，并在CASP14结束后又经过一些调整，并发布（开源）于2021年7月。为区别于Deepmind团队在此前开发的另一个用于预测蛋白质结构的一个人工智能模型“Alphafold”，这里我们称之为“Alphafold 2”。
> 本文主要针对Alphafold 2进行介绍，而非Alphafold。
> 作者在学习相关文献*Highly accurate protein structure prediction with AlphaFold*及其补充材料后，根据作者对此模型的理解，总结成文。**必须强调的是，作者对人工智能以及蛋白结构领域的了解都较为浅薄，本文仅根据作者个人的理解总结成文，并不能保证内容的绝对正确，更不能用作学习Alphafold2模型的参考。**
> 本文最重要的参考文献即为[*Highly accurate protein structure prediction with AlphaFold*](https://doi.org/10.1038/s41586-021-03819-2)。本文中若参考此文献则不再进行标注。
> 本文的网页版详见此处[Alphafold 2 算法解析]()（贴网页）。

---

### 简介

生命体中各种各样的生命过程大多由蛋白质参与完成，而对蛋白质的结构进行解析即使立足于当下的技术手段，依旧是一件具有挑战性的事情。经管如此，经过一代代结构生物学家的不懈努力，人们也已经通过实验的手段确定了超过100,000种独特蛋白质的结构，这为人工智能模型的构建提供了可行的训练数据。

根据蛋白质的一级结构预测其三级结构，即根据蛋白质种的氨基酸序列预测蛋白质3D结构，一直是一个困难但重要的科学问题。对这一问题的解决主要有两种策略，一种策略基于物理相互作用，考虑原子-原子之间的相互作用，利用热力学和动力学模拟来进行蛋白质结构的预测，尽管这种接近”第一性计算“的策略在理论上非常有说服力，但大分子模拟的难度依旧太高；另一种策略基于生物演化过程，利用未知结构与已知结构的同源性和演化过程中的保守性，基于已知结构来预测未知结构，但这种方法依赖于同源序列的结构，但在大多数情况下，一个蛋白质的同源物的结构同样也是未知的。在Alphafold2出现之前，各种各样的预测方法给出的大多数预测准确性远低于实验所得结构的准确性。

Alphafold 2是一个用于预测蛋白质3D结构的神经网络模型，该模型于2020年参加CASP14，并取得了惊人的结果。在 CASP14 中，AlphaFold 2结构的骨架准确度中位数为 0.96 Å r.m.s.d._95_（95% 残基覆盖率时的 Cα 均方根偏差）（95% 置信区间 = 0.85–1.16 Å）AlphaFold 的全原子精度为 1.5 Å r.m.s.d._95_（95% 置信区间 = 1.2-1.6 Å）。这样的结果远比同时参加CASP14的其他方法优秀，甚至在一定程度上可以和实验解得的结构相媲美。

---

### Alphafold 2框架概览

![AF1](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF1.png?raw=true)

Alphafold 2主要实现的功能是从一个蛋白质的氨基酸序列预测蛋白质的3D结构，换句话说，该模型需要一个**输入：一条氨基酸序列**；并利用这一输入产生一个**输出：蛋白质的3D结构**`Fig. 1 绿色部分`。这个3D结构有如下两个特点：
	分辨率是原子级别（即可以预测每个重原子所在的具体位置）
	端到端3D结构预测（即肽链从N端到C端的完整结构都将被预测）
	该3D结构仅属于一个肽链（即不能同时对complex中的各个结构进行预测）
	该模型不从物理上考虑金属离子与蛋白的互作等（有趣的是，在经过大量结构的学习之后，Alphafold 2在一定程度上能够预测蛋白与金属离子的作用，即使该模型并不从物理相互作用上考虑这一特点）

输入的这一条氨基酸序列通过在序列数据库中通过多序列比对（multiple sequence alignment, MSA）来查找同源序列，并进一步组建出**MSA representation**；同时，通过在结构数据库中查找同源区段以及Pairing的过程组建**Pair representation**。MSA representation携带有输入序列和其同源序列的同源信息（演化相关的信息）以及大量其他序列的feature；Pair representation携带有输入序列的任意两个氨基酸残基之间的相关性信息（初始状态）。`Fig. 1 蓝色部分`

在这之后MSA representation以及Pair representation作为**Evoformer网络**的输入，在Evoformer网络中不断交换演化相关的信息和氨基酸残基之间的相关性信息，并最终输出一个经过若干次**更新（update）后的MSA representation和Pair representation**。`Fig. 1 橙色部分`

从Evoformer网络中输出的MSA representation的第一行即为原始输入序列的信息；从Evoformer网络中输出的Pair representation中携带了更新后的任意氨基酸之间的相关性信息。上述两者作为**Structure module**的输入，在Stucture module中进行蛋白质结构的构建。`Fig. 1 紫色部分`，并最终输出预测得到的蛋白质中各重原子的3D结构以及各项评估参数（如，plDDT-Cα等）`Fig. 1 右侧绿色部分`

从Evoformer网络中输出的MSA representation的第一行以及Pair representation同样还会再与蓝色部分所得相融合（embedding）再作为Evoformer的输入经历上述过程，这一机制被称为**Recycling**，将会进行3次Recycling。`Fig. 1 红色部分`

### Input embeddings

### Evoformer blocks

### Structure module

### Recycling

### Database

### Training schema