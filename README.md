# Awesome Deep Community Detection (References continually updated)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A collection of papers on deep learning for community detection.

- [Awesome Deep Community Detection](#awesome-Deep-Community-Detection)
  - [Survey](#survey)
  - [Deep Neural Network-based Community Detection](#deep-neural-network-based-community-detection)
  	- [CNN-based Approaches](#CNN-based-approaches)
  	- [Auto-encoder-based Approaches](#auto-encoder-based-approaches)
  	- [GAN-based Approaches](#GAN-based-approaches)
  - [Deep Graph Embedding-based Community Detection](#deep-graph-embedding-based-community-detection)
	- [Deep NMF-based Approaches](#deep-NMF-based-approaches)
	- [Deep SF-based Approaches](#deep-SF-based-approaches)
	- [Community Embedding-based Approaches](#community-embedding-based-approaches)
  - [Graph Neural Network-based Community Detection](#graph-neural-network-based-community-detection)


## Survey
__Deep Learning for Community Detection: Progress, Challenges and Opportunities__. Preprint IJCAI 2020. _Fanzhen Liu, Shan Xue, Jia Wu, Chuan Zhou, Wenbin Hu, Cecile Paris, Surya Nepal, Jian Yang, Philip S. Yu_. [[Paper](https://arxiv.org/abs/2005.08225)] [[AI科技评论](https://cloud.tencent.com/developer/article/1632305)]

__Community Detection in Networks: A Multidisciplinary Review__. Journal of Network and Computer Applications 2018. _Muhammad Aqib Javed, Muhammad Shahzad Younis, Siddique Latif, Junaid Qadir, Adeel Baig_. [[Paper](https://www.sciencedirect.com/science/article/pii/S1084804518300560)]

__Community Discovery in Dynamic Networks: A Survey__. ACM Computing Surveys 2018. _Giulio Rossetti, Rémy Cazabet_. [[Paper](https://dl.acm.org/doi/10.1145/3172867)]

__Metrics for Community Analysis: A Survey__. ACM Computing Surveys 2017. _Tanmoy Chakraborty, Ayushi  Dalmia, Ayushi Dalmia, Animesh  Mukherjee, Animesh Mukherjee, Niloy Ganguly_. [[Paper](https://dl.acm.org/doi/10.1145/3091106)]

__Network Community Detection: A Review and Visual Survey__. Preprint 2017. _Bisma S. Khan, Muaz A. Niazi_. [[Paper](https://arxiv.org/abs/1708.00977)]

__Community Detection: A User Guide__. Physics Reports 2016. _Santo Fortunato, Darko Hric_. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0370157316302964)]

__Community Detection in Social Networks__. WIREs Data Mining Knowledge Discovery 2016. _Punam Bedi, Chhavi Sharma_. [[Paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/widm.1178)]


## Deep Neural Network-based Community Detection
### CNN-based Approaches
__A deep learning based community detection approach__. SAC 2019. _Giancarlo Sperlí_. [[Paper](https://doi.org/10.1145/3297280.3297574)]

__Deep community detection in topologically incomplete networks__. Physica A 2017. _Xin et al._ [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378437116308342)]

### Auto-encoder-based Approaches
__Learning community structure with variational autoencoder__. ICDM 2018. _Choong et al_. [[Paper](https://ieeexplore.ieee.org/document/8594831)]

__Deep network embedding for graph representation learning in signed networks__. IEEE TCYB 2018. _Xiao Sheng and Fu-Lai Chung_. [[Paper](https://ieeexplore.ieee.org/document/8486671)] [[Code](https://github.com/shenxiaocam/Deep-network-embedding-for-graph-representation-learning-in-signed-networks)]

__Incorporating network structure with node contents for community detection on large networks using deep learning__. Neurocomputing 2018. _Cao et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231218300985)]	

__Autoencoder based community detection with adaptive integration of network topology and node contents__. KSEM 2018. _Cao et al._. [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-99247-1_16)]

__DFuzzy: A deep learning-based fuzzy clustering model for large graphs__. Knowledge and Information Systems 2018. _Vandana Bhatia and Rinkle Rani_. [[Paper](https://link.springer.com/article/10.1007/s10115-018-1156-3)]

__Modularity based community detection with deep learning__. IJCAI 2016. _Yang et al._. [[Paper](https://www.ijcai.org/Proceedings/16/Papers/321.pdf)] [[Code](http://yangliang.github.io/code/DC.zip)]

### GAN-based Approaches
__CommunityGAN: Community detection with generative adversarial
nets__. WWW 2019. _Jia et al._. [[Paper](https://dl.acm.org/doi/abs/10.1145/3308558.3313564)] [[Code](https://github.com/SamJia/CommunityGAN)]

__Learning graph representation with generative adversarial nets__. IEEE TKDE 2019. _Wang et al._. [[Paper](https://ieeexplore.ieee.org/document/8941296)] [[Code](https://github.com/hwwang55/GraphGAN)]


## Deep Graph Embedding-based Community Detection
### Deep NMF-based Approaches
__Deep autoencoder-like nonnegative matrix factorization for community detection__. CIKM 2018. _Ye et al._. [[Paper](https://dl.acm.org/doi/10.1145/3269206.3271697)] [[Code](https://github.com/benedekrozemberczki/DANMF)]

__Community detection in attributed graphs: An embedding approach__. AAAI 2018. _Li et al._. [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17142/15705)]

### Deep SF-based Approaches
__Community discovery in networks with deep sparse filtering__. Pattern Recognition 2018. _Xie et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S003132031830116X)]

### Community Embedding-based Approaches
__A unified framework for community detection and network representation learning__. IEEE TKDE 2019. _Tu et al._. [[Paper](https://ieeexplore.ieee.org/document/8403293)] [[Code](http://nlp.csai.tsinghua.edu.cn/~tcc/datasets/simplified_CNRL.zip)]

__Cosine: Community-preserving social network embedding from information
diffusion cascades__. AAAI 2018. _Zhang et al._. [[Paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16364/159824)]

__Learning community embedding with community detection and node embedding on graphs__. CIKM 2017. _Cavallari et al._. [[Paper](https://dl.acm.org/doi/10.1145/3132847.3132925)] [[Code](https://github.com/vwz/ComE)]

## Graph Neural Network-based Community Detection
__Supervised community detection with line graph neural networks__. ICLR 2019. _Chen et al._. [[Paper](https://openreview.net/pdf?id=H1g0Z3A9Fm)] [[Code](https://github.com/zhengdao-chen/GNN4CD)]

__Overlapping community detection with graph neural networks__. Deep Learning on Graphs Workshop, KDD 2019. _Oleksandr Shchur and Stephan Günnemann_. [[Paper](https://doi.org/10.1145/3297280.3297574)] [[Code](https://github.com/shchur/overlapping-community-detection)]

__Graph convolutional networks meet markov random fields: Semi-supervised community detection in attribute networks__. AAAI 2019. _Jin et al._. [[Paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3780/3658)]
