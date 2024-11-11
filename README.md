# Domain-Consensus-Clustering-CDD
I modified the code of the 2020 CVPR paper Domain Consensus Clustering for Universal Domain Adaptation. The original CDD loss in the code only applied to samples already clustered as common categories. However, based on the CDD design principle—which aims to pull samples of the same category closer together and push samples of different categories further apart—I adjusted the loss calculation to include categories not initially clustered as common. The results, stored in the result.txt file for the Office31 dataset, showed improved performance.
