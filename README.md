# AlphaFold2 算法解析

傅乾正

---

> #### 写在前面
> 
> 本文中所述的“AlphaFold2”指Deepmind团队开发的一个用于预测蛋白质结构的一个神经网络模型，该模型在CASP14中展现出了碾压性的优势，并在CASP14结束后又经过一些调整，并发布（开源）于2021年7月。为区别于Deepmind团队在此前开发的另一个用于预测蛋白质结构的一个人工智能模型“AlphaFold”，这里我们称之为“AlphaFold2”。
> 
> 本文主要针对AlphaFold2进行介绍，而非Alphafold。
> 
> 作者在学习相关文献*Highly accurate protein structure prediction with AlphaFold*及其补充材料后，根据作者对此模型的理解，总结成文。**必须强调的是，作者对人工智能以及蛋白结构领域的了解都较为浅薄，本文仅根据作者个人的理解总结成文，并不能保证内容的绝对正确，更不能用作学习Alphafold2模型的参考。**
> 
> 本文最重要的参考文献即为[*Highly accurate protein structure prediction with AlphaFold*](https://doi.org/10.1038/s41586-021-03819-2)。本文中此参考文不再进行标注。
> 
> 本文的网页版详见此处[AlphaFold2 算法解析](https://github.com/shijiu001/Alphafold2_ideas/blob/main/%23%20AlphaFold2%20%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90.md)，可编辑版详见此处[RawMD](https://github.com/shijiu001/Alphafold2_ideas/blob/main/%23%20AlphaFold2%20%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90.md?raw=true)。

---

### 简介

生命体中各种各样的生命过程大多由蛋白质参与完成，而对蛋白质的结构进行解析即使立足于当下的技术手段，依旧是一件具有挑战性的事情。经管如此，经过一代代结构生物学家的不懈努力，人们也已经通过实验的手段确定了超过100,000种独特蛋白质的结构，这为人工智能模型的构建提供了可行的训练数

根据蛋白质的一级结构预测其三级结构，即根据蛋白质种的氨基酸序列预测蛋白质3D结构，一直是一个困难但重要的科学问题。对这一问题的解决主要有两种策略，一种策略基于物理相互作用，考虑原子-原子之间的相互作用，利用热力学和动力学模拟来进行蛋白质结构的预测，尽管这种接近”第一性计算“的策略在理论上非常有说服力，但大分子模拟的难度依旧太高；另一种策略基于生物演化过程，利用未知结构与已知结构的同源性和演化过程中的保守性，基于已知结构来预测未知结构，但这种方法依赖于同源序列的结构，但在大多数情况下，一个蛋白质的同源物的结构同样也是未知的。在Alphafold2出现之前，各种各样的预测方法给出的大多数预测准确性远低于实验所得结构的准确性。

AlphaFold2是一个用于预测蛋白质3D结构的神经网络模型，该模型于2020年参加CASP14，并取得了惊人的结果。在 CASP14 中，AlphaFold2结构的骨架准确度中位数为 0.96 Å r.m.s.d.<sub>95</sub>（95% 残基覆盖率时的 Cα 均方根偏差）（95% 置信区间 = 0.85–1.16 Å）AlphaFold 的全原子精度为 1.5 Å r.m.s.d.<sub>95</sub>（95% 置信区间 = 1.2-1.6 Å）。这样的结果远比同时参加CASP14的其他方法优秀，甚至在一定程度上可以和实验解得的结构相媲美。

---

### AlphaFold2框架概览

![AF1](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF1.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Fig. 1e</center> 

AlphaFold2主要实现的功能是从一个蛋白质的氨基酸序列预测蛋白质的3D结构，换句话说，该模型需要一个**输入：一条氨基酸序列**；并利用这一输入产生一个**输出：蛋白质的3D结构**`Fig. 1e 绿色部分`。这个3D结构有如下两个特点：
	分辨率是原子级别（即可以预测每个重原子所在的具体位置）
	端到端3D结构预测（即肽链从N端到C端的完整结构都将被预测）
	该3D结构仅属于一个肽链（即不能同时对complex中的各个结构进行预测）
	该模型不从物理上考虑金属离子与蛋白的互作等（有趣的是，在经过大量结构的学习之后，AlphaFold2在一定程度上能够预测蛋白与金属离子的作用，即使该模型并不从物理相互作用上考虑这一特点）

输入的这一条氨基酸序列通过在序列数据库中通过多序列比对（multiple sequence alignment, MSA）来查找同源序列，并进一步组建出**MSA representation**；同时，通过在结构数据库中查找同源区段以及Pairing的过程组建**Pair representation**。MSA representation携带有输入序列和其同源序列的同源信息（演化相关的信息）以及大量其他序列的feature；Pair representation携带有输入序列的任意两个氨基酸残基之间的相关性信息（初始状态）。`Fig. 1e 蓝色部分`

在这之后MSA representation以及Pair representation作为**Evoformer网络**的输入，在Evoformer网络中不断交换演化相关的信息和氨基酸残基之间的相关性信息，并最终输出一个经过若干次**更新（update）后的MSA representation和Pair representation**。`Fig. 1e 橙色部分`

从Evoformer网络中输出的MSA representation的第一行即为原始输入序列的信息；从Evoformer网络中输出的Pair representation中携带了更新后的任意氨基酸之间的相关性信息。上述两者作为**Structure module**的输入，在Stucture module中进行蛋白质结构的构建。`Fig. 1e 紫色部分`，并最终输出预测得到的蛋白质中各重原子的3D结构以及各项评估参数（如，plDDT-Cα等）`Fig. 1e 右侧绿色部分`

从Evoformer网络中输出的MSA representation的第一行以及Pair representation同样还会再与蓝色部分所得相整合（embedding）再作为Evoformer的输入经历上述过程，这一机制被称为**Recycling**，将会进行3次Recycling。`Fig. 1e 红色部分`

---

### Input embeddings

![AF2](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF2.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">SFig. 1</center> 

正如上文所述，AlphaFold2需要首先组建MSA representation以及Pair representation，本节就先介绍MSA representation以及Pair representation如何组建以及集成了哪些特征。

#### 首次组建MSA representation以及Pair representation

模型运行伊始，尚未产生能够用于Recycling的数据，因此，首次组建MSA representation以及Pair representation的过程与之后Recycling中重新组建MSA representation以及Pair representation的过程略有不同。SFig. 1中红色部分展示了首次组建MSA representation以及Pair representation的过程。

AlphaFold2首先会将输入的序列以及从数据库中查找到的诸多信息整合为6个重要的feature：target_feat, residue_index, msa_feat, extra_msa_feat, template_pair_feat, template_angle_feat。在得到上述的6个feature后，对其中三个进行一些列的线性变换（更严谨的讲，这里也经过了其他一些变换过程，这一过程本质就是初步的特征提取）并最终得到MSA representation以及Pair representation。

> 具体每个feature中都整合了哪些信息在文章的[补充材料](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf)中有详细的解释。因其对后文的理解并不起关键作用，本文不再赘述这部分内容。

值得一提的是，对于residue_index的处理使用了一个被称为”relpos“的函数，这个函数将每个氨基酸残基视为原点，将其前后分别32个氨基酸残基归入一个bin中，这意味着对于任意一个氨基酸残基而言，其被考虑的相互作用残基被限制在了前后32个残基以内，这大大降低了计算难度。

> 这里并未直接阻止模型成功预测超过32个残基的”远距离相互作用“，因为这种相互作用依旧可以通过几个中间残基作为桥梁而得到建立。

在组建MSA representation以及Pair representation之后，Alphafold会进一步将得到的template_pair_feat整合到Pair representation中；template_angle_feat整合到MSA representation中；将extra_msa_feat整合为一个extra_MSA representation。

#### Recycling中的MSA representation以及Pair representation组建

上文中已经提到，进行Recycling的数据包括Evoformer网络中输出的MSA representation的第一行以及Pair representation。此时，Alphafold会整合一套之前没有被整合在MSA representation中的同源序列得到一组新的MSA representation。这一过程可以认为是引入了新的同源序列信息，而之所以不在第一次组建MSA representation时就使用全部的同源序列，是因为那样的计算成本太高。

MSA representation得到更新，Pair representation保持不变，此时只需要进行红色部分之外的其他部分的整合即可。

---

### Evoformer blocks

![AF3](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF3.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Fig. 3a</center> 

在得到了MSA representation以及Pair representation后，它们便会作为输入进入Evoformer网络。此网络的结构如Fig. 3a所示。Evoformer网络用来计算MSA representation和Pair representation。通过它们的不断更新，来充分交换MSA representation携带的进化信息给Pair representation。具体的说，信息的交换和更新主要是通过一系列的Attention层来完成的，下文将对Evoformer的过程进行介绍。

> Attention机制是近年来自然语言处理领域广泛应用的一种机制，其核心思想是加权处理，对于当前的序列，根据其不同的重要程度，赋予不同的权重，以实现关注/处理更重要信息的能力。其中最重要的参数为：Query，Key，Value。其过程简单而言就是将Query和Key作用得到的attention权值作用到Value上，其中也可以进一步引入bias。

![AF4](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF4.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">SFig. 2</center> 

首先是**对MSA representation按行进行的self-attention**，在此过程中，利用Pair representation计算得到一个bias引入self-attention的过程。实现了两者之间的信息交换。按行进行的attention使得每个同源序列中重要的那部分序列得到凸显（我们知道MSA中引入的同源序列大概率只有部分片段同源，因此这一过程使得同源部分得到了凸显）。

![AF5](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF5.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">SFig. 3</center> 

然后是**对MSA representation按列进行的self-attention**。这一过程并未引入更多bias，实现了对同源序列相同位置的突变和保守的识别（我们知道MSA引入的同源序列在同一氨基酸位置处并非条条都同源，因此这一过程使得某一位置处同源的序列得到了凸显，当然，一些保守性突变也应该得到凸显）。

![AF6](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF6.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">SFig. 4</center> 

值得注意的是上述过程并未直接对原有的MSA representation进行更新，因此此时的空间以不同与之前，需要relu函数进行空间映射，这一步被称为**Transition**。

> 上述过程与Transformer模型的前馈网络神似，猜想这就是Evoformer名称的由来吧。

![AF7](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF7.png?raw=true)

接下来是一次比较关键的信息交换，利用**Transition**得到的MSA representation，对其中的每两列进行进行变换（改变其维度，提取其特征）张量积（outer product mean）在对其按列取均值（即对s所代表的维度取均值，参考图中标注的维度变化则容易理解），之后再进行一次线性变换得到一个氨基酸残基对之间的相关性信息。这一步充分的将MSA representation中携带的信息交换给了pair representation。这一步称为**Outer product mean**。

![AF8](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF8.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Fig. 3b, 3c</center> 

之后是Pair representation的更新，称为**Triangular multiplicative update**。其基本逻辑是，这个矩阵描述的是任意两个氨基酸残基之间的关系，这些关系不是毫无约束的。比如距离，距离关系不是自由的，应该满足比如三角不等式，相邻的三个边应该满足两边之和大于第三边。所以这里的Pair representation的更新使用了**Triangular multiplicative update**。对于每个边，都会接收到和他组成三角形的任意两个其他边带来的更新，分为内和外（"incoming" & "outgoing"）两个过程。

之后是两层按行进行的**Triangular self-attention**，分别被称作“around starting node”和”around ending node“，和之前的"incoming" & "outgoing"的概念类似。按行进行self-attention，则可以再诸多氨基酸对的相互作用中找到较为重要的那些。

再之后是一次**Transition**，这里的Transtion与之前提到的Transition几乎一样，不再赘述。

上述过程就是一个Evoformer block，重复上述过程48次，我们就完成了Evoformer blocks的任务`Fig. 1e 橙色部分`.

---

### Recycling

Evoformer网络最终输出的MSA representation的第一行以及Pair representation会进行Recycling，前文已经有过介绍，这里不再赘述。

---

### Structure module

![AF9](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF9.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Fig. 3d</center> 

上文中已经提到，Evoformer网络最终输出的MSA representation的第一行（被称为Single repr.）以及Pair representation会作为Structure module的输入进入这一模块。Structure module的主要功能是将蛋白质结构的抽象特征（即Sing repr.和Pair representation）映射成为具体的原子坐标。这一模块有8个共享权重的blocks。这一模块的主要输出有两个部分，一是蛋白质的3D结构，另一是对该预测结构的评价。

Structure module对蛋白质结构的初始化被称为”黑洞初始化“（”black hole initialization“）,这种初始化方法将所有的氨基酸残基初始化在一点上（原点），并且都初始化为同一方向。值得注意的是，这一初始化是针对backbone而言的。

> 所谓backbone之预测结构中的骨架信息，也可以简单理解为由（氨基N-Cα-羧基C）构成的主链。backbone规定了每个氨基酸残基的位置及角度。

![AF12](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF12.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">SFig. 8</center> 

Single repr.，Pair representation，backbone作为**Invariant Point Attention Module (IPA)**的输入，在IPA中经过一系列Attention层的处理，产生一个被更新的Update.。这个过程中，Pair representation会计算产生一个bias被用于Attention，已经产生的backbone信息（这里主要针对第一次循环之后的部分，第一次进入IPA是，backbone信息是”黑洞初始化“的状态）也会经过Attention层处理。这些信息最终会整合成为一个被更新的Update。

这个被更新的Update，在**Predict relative rotations and transtations**中用于更新backbone。这一更新过程通过预测各残基之间的旋转和平移来实现。特别地，在AlphaFold2模型中，旋转的表示使用四元数（quaternion）来完成，平移的表示使用向量来完成。

> 值得注意的是，在AlphaFold2模型中，在预测时对会将每个氨基酸残基放在原点，然后预测其前后氨基酸的相对位置。因此为了还原整体的3D结构，就必须记录每个氨基酸残基为原点的坐标系向整体蛋白质的坐标系还原的欧几里得变换信息。这样做对于机器学习由巨大的好处，这使得机器学习得到的信息能够轻松的用于每个氨基酸残基上。

![AF10](https://github.com/shijiu001/Alphafold2_ideas/blob/main/AF10.png?raw=true)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Fig. 3e</center> 

在backbone预测完成后，为预测每个原子所在的位置，还需要对侧链的扭转角进行预测。每个氨基酸残基的的原子可以表示为S<sub>atom names</sub> = {N, Cα, C, O, Cβ, Cγ, Cγ1, Cγ2, . . . }，而其间的扭转角则表示为 S<sub>torsion names</sub> = {ω, Φ, Ψ, χ1, χ2, χ3, χ4}。由于主链已经得到确定，因此ω和Φ将不会在这个过程中被改变。对其他扭转角的确定则基于ResNet预测和物理规律的规范。其中物理规律的规范主要通过计算损失（losses）以及在机器学习过程中引导梯度下降来完成。最后，为了解决剩余的一些和物理规律不符的点，模型中引入了Amber relaxation机制。这一机制将不符合物理规律的残基上的约束去除，并利用物理模型对其重新进行约束。

最终Structure module会输出3D结构（算法上讲，3D结构用一个向量集表示），以及4项评估参数。

---
