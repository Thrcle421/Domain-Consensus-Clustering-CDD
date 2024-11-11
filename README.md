# Domain-Consensus-Clustering-CDD
对2020年CVPR Domain Consensus Clustering for Universal Domain Adaptation论文源码做了修改，其中使用的CDD loss只是针对其已经聚类分为common的样本类别，但是根据CDD的设计原理，
是将相同类别的样本拉近，不同类别的样本推远，根据这样的想法，将其未分为common的类别加入了loss的计算中。
在office31数据集中的结果在result.txt文件中，达到了更好的效果。
