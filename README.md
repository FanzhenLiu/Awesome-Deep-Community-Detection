# Awesome Deep Community Detection
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A collection of papers on deep learning for community detection.

- [Awesome Deep Community Detection](#awesome-Deep-Community-Detection)
  - [Survey](#survey)
  - [Convolutional Networks-based Community Detection](#convolutional-networks-based-community-detection)
  	- [CNN-based Community Detection](#cnn-based-community-detection)
  	- [GCN-based Community Detection](#gcn-based-community-detection)
  - [Graph Attention Network-based Community Detection](#graph-attention-network-based-community-detection)
  - [Graph Adversarial Network-based Community Detection](#graph-adversarial-network-based-community-detection)
  - [Autoencoder-based Community Detection](#autoencoder-based-community-detection)
	- [Stacked AE-based Community Detection](#stacked-ae-based-community-detection)
	- [Sparse AE-based Community Detection](#sparse-ae-based-community-detection)
	- [Denoising AE-based Community Detection](#denoising-ae-based-community-detection)
	- [Graph Convolutional AE-based Community Detection](#graph-convolutional-ae-based-community-detection)
	- [Graph Attention AE-based Community Detection](#graph-attention-ae-based-community-detection)
	- [Variational AE-based Community Detection](#variational-ae-based-community-detection)
  - [Deep Nonnegative Matrix Factorization-based Community Detection](#deep-nonnegative-matrix-factorization-based-community-detection)
  - [Deep Sparse Filtering-based Community Detection](#deep-sparse-filtering-based-community-detection)
  - [Datasets](#datasets)
  - [Tools](#tools)


## Survey
__A Comprehensive Survey on Community Detection with Deep Learning__. **28 Pages**, arXiv 2021. _Xing Su, Shan Xue, Fanzhen Liu, Jia Wu, Jian Yang, Chuan Zhou, Wenbin Hu, Cecile Paris, Surya Nepal, Di Jin, Quan Z. Sheng, Philip S. Yu_. [[Paper](https://arxiv.org/pdf/2105.12584.pdf)] 

Link: https://arxiv.org/abs/2105.12584

    @inproceedings{su2021survey,
    	author = {Su, Xing and Xue, Shan and Liu, Fanzhen and Wu, Jia and Yang, Jian and 
		Zhou, Chuan and Hu, Wenbin and Paris, Cecile and Nepal, Surya and Jin, Di and 
		Sheng, Quan Z. and Yu, Philip S.},
		eprint={2105.12584},
		archivePrefix={arXiv},
    	title = {A Comprehensive Survey on Community Detection with Deep Learning},
    	year = {2021},
    }

__Deep Learning for Community Detection: Progress, Challenges and Opportunities__. IJCAI 2020. _Fanzhen Liu, Shan Xue, Jia Wu, Chuan Zhou, Wenbin Hu, Cecile Paris, Surya Nepal, Jian Yang, Philip S. Yu_. [[Paper](https://www.ijcai.org/Proceedings/2020/0693.pdf)] [[AI科技评论](https://cloud.tencent.com/developer/article/1632305)]

Link: https://www.ijcai.org/Proceedings/2020/693

    @inproceedings{ijcai2020-693,
    	author = {Liu, Fanzhen and Xue, Shan and Wu, Jia and Zhou, Chuan and Hu, Wenbin and 
		Paris, Cecile and Nepal, Surya and Yang, Jian and Yu, Philip S.},
    	booktitle  = {Proceedings of the Twenty-Ninth International Joint Conference on
		Artificial Intelligence, {IJCAI-20}},
    	title = {Deep Learning for Community Detection: Progress, Challenges and Opportunities},
    	year = {2020},
		pages = {4981-4987},
		doi = {10.24963/ijcai.2020/693}
    }

__Community Detection in Networks: A Multidisciplinary Review__. Journal of Network and Computer Applications 2018. _Muhammad Aqib Javed, Muhammad Shahzad Younis, Siddique Latif, Junaid Qadir, Adeel Baig_. [[Paper](https://www.sciencedirect.com/science/article/pii/S1084804518300560)]

__Community Discovery in Dynamic Networks: A Survey__. ACM Computing Surveys 2018. _Giulio Rossetti, Rémy Cazabet_. [[Paper](https://dl.acm.org/doi/10.1145/3172867)]

__Metrics for Community Analysis: A Survey__. ACM Computing Surveys 2017. _Tanmoy Chakraborty, Ayushi  Dalmia, Ayushi Dalmia, Animesh  Mukherjee, Animesh Mukherjee, Niloy Ganguly_. [[Paper](https://dl.acm.org/doi/10.1145/3091106)]

__Network Community Detection: A Review and Visual Survey__. Preprint 2017. _Bisma S. Khan, Muaz A. Niazi_. [[Paper](https://arxiv.org/abs/1708.00977)]

__Community Detection: A User Guide__. Physics Reports 2016. _Santo Fortunato, Darko Hric_. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0370157316302964)]

__Community Detection in Social Networks__. WIREs Data Mining Knowledge Discovery 2016. _Punam Bedi, Chhavi Sharma_. [[Paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/widm.1178)]


## Convolutional Networks-based Community Detection
### CNN-based Community Detection
__Deep community detection in topologically incomplete networks__. Physica A 2017. _Xin et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378437116308342)]

__A deep learning based community detection approach__. SAC 2019. _Giancarlo Sperlí_. [[Paper](https://doi.org/10.1145/3297280.3297574)]

__Edge classification based on convolutional neural networks for community detection in complex network__. Physica A 2020. _Cai et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378437120304271)]

### GCN-based Community Detection 
__Supervised community detection with line graph neural networks__. ICLR 2019. _Chen et al._. [[Paper](https://openreview.net/pdf?id=H1g0Z3A9Fm)] [[Code](https://github.com/zhengdao-chen/GNN4CD)]

__Graph convolutional networks meet markov random fields: Semi-supervised community detection in attribute networks__. AAAI 2019. _Jin et al._. [[Paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3780/3658)]

__CommDGI: Community Detection Oriented Deep Graph Infomax__. CIKM 2020. _Zhang et al._. [[Paper](https://dl.acm.org/doi/abs/10.1145/3340531.3412042)] 

__Overlapping community detection with graph neural networks__. Deep Learning on Graphs Workshop, SIGKDD 2019. _Oleksandr Shchur and Stephan Günnemann_. [[Paper](https://deep-learning-graphs.bitbucket.io/dlg-kdd19/accepted_papers/DLG_2019_paper_3.pdf)] [[Code](https://github.com/shchur/overlapping-community-detection)]

__Going deep: Graph convolutional ladder-shape networks__. AAAI 2020. _Hu et al._. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5673/5529)]

__Independence promoted graph disentangled networks__. AAAI 2020. _Liu et al._. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5929/5785)]

__Attributed graph clustering via adaptive graph convolution__. IJCAI 2019. _Zhang et al._. [[Paper](https://www.ijcai.org/Proceedings/2019/0601.pdf)] [[Code](https://github.com/karenlatong/AGC-master)]

__Adaptive graph encoder for attributed graph embedding__. SIGKDD 2020. _Cui et al._. [[Paper](https://dl.acm.org/doi/10.1145/3394486.3403140)] [[Code](https://github.com/thunlp/AGE)]

__CayleyNets: Graph convolutional neural networks with complex rational spectral filters__.  IEEE Transactions on Signal Processing 2019. _Levie et al._. [[Paper](https://ieeexplore.ieee.org/document/8521593)] [[Code](https://github.com/amoliu/CayleyNet)]


## Graph Attention Network-based Community Detection
__Unsupervised Attributed Multiplex Network Embedding__. AAAI 2020. _Park et al._. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5985)]

__MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding__. WWW 2020. _Fu et al._. [[Paper](https://dl.acm.org/doi/abs/10.1145/3366423.3380297)]


## Graph Adversarial Network-based Community Detection
__SEAL: Learning Heuristics for Community Detection with Generative Adversarial Networks__. KDD 2020. _Zhang et al._. [[Paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403154)] 

__Multi-Class Imbalanced Graph Convolutional Network Learning__. IJCAI 2020. _Shi et al._. [[Paper](https://www.ijcai.org/proceedings/2020/398)]

__JANE: Jointly adversarial network embedding__. IJCAI 2020. _Yang et al._. [[Paper](https://www.ijcai.org/Proceedings/2020/0192.pdf)]

__ProGAN: Network Embedding via Proximity Generative Adversarial Network__. KDD 2019. _Gao et al._. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3292500.3330866)]

__CommunityGAN: Community detection with generative adversarial nets__. WWW 2019. _Jia et al._. [[Paper](https://dl.acm.org/doi/abs/10.1145/3308558.3313564)] [[Code](https://github.com/SamJia/CommunityGAN)]


## Autoencoder-based Community Detection
### Stacked AE-based Community Detection

__Modularity based community detection with deep learning__. IJCAI 2016. _Yang et al._. [[Paper](https://www.ijcai.org/Proceedings/16/Papers/321.pdf)] [[Code](http://yangliang.github.io/code/DC.zip)]

__Deep network embedding for graph representation learning in signed networks__. IEEE TCYB 2018. _Xiao Sheng and Fu-Lai Chung_. [[Paper](https://ieeexplore.ieee.org/document/8486671)] [[Code](https://github.com/shenxiaocam/Deep-network-embedding-for-graph-representation-learning-in-signed-networks)]

__Integrative network embedding via deep joint reconstruction__. IJCAI 2018. _Hongchang Gao and Heng Huang_. [[Paper](https://www.ijcai.org/Proceedings/2018/0473.pdf)]

__An evolutionary autoencoder for dynamic community detection__. Science China Information Sciences 2020. _Wang et al._. [[Paper](https://link.springer.com/article/10.1007/s11432-020-2827-9)]

__Deep attributed network embedding__. IJCAI 2018. _Gao et al._. [[Paper](https://www.ijcai.org/Proceedings/2018/0467.pdf)] [[Code](https://github.com/gaoghc/DANE)]

__High-performance community detection in social networks using a deep transitive autoencoder__. Information Sciences 2019. _Xie et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025519303251)]

__BL-MNE: Emerging heterogeneous social network embedding through broad learning with aligned autoencoder__. ICDM 2017. _Zhang et al._. [[Paper](https://doi.org/10.1109/ICDM.2017.70)] [[Code](http://www.ifmlab.org/files/code/Aligned-Autoencoder.zip)]

### Sparse AE-based Community Detection

__Learning deep representations for graph clustering__. AAAI 2014. _Tian et al._. [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8527/8571)] [[Code](https://github.com/quinngroup/deep-representations-clustering)]

__DFuzzy: A deep learning-based fuzzy clustering model for large graphs__. Knowledge and Information Systems 2018. _Vandana Bhatia and Rinkle Rani_. [[Paper](https://link.springer.com/article/10.1007/s10115-018-1156-3)]

__Stacked autoencoder-based community detection method via an ensemble clustering framework__. Information Sciences 2020. _Xu et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S002002552030270X)]

### Denoising AE-based Community Detection

__MGAE: Marginalized graph autoencoder for graph clustering__. CIKM 2017. _Wang et al._. [[paper](https://dl.acm.org/doi/10.1145/3132847.3132967)] [[Code](https://github.com/FakeTibbers/MGAE)]

__Deep neural networks for learning graph representations__. AAAI 2016. _Cao et al._. [[paper](https://dl.acm.org/doi/10.5555/3015812.3015982)]

__Graph clustering with dynamic embedding__. arXiv. Yang et al._. [[paper](https://arxiv.org/abs/1712.08249)] 

### Graph Convolutional AE-based Community Detection

__Community-centric graph convolutional network for unsupervised community detection__. IJCAI 2020. _He et al._. [[Paper](https://www.ijcai.org/Proceedings/2020/0486.pdf)]

__Structural deep clustering network__.  WWW 2020. _Bo et al._. [[Paper](https://dl.acm.org/doi/10.1145/3366423.3380214)] [[Code](https://github.com/bdy9527/SDCN)]

__One2Multi graph autoencoder for multi-view graph clustering__. WWW 2020. _Fan et al._. [[Paper](https://dl.acm.org/doi/10.1145/3366423.3380079)] [[Code](https://github.com/googlebaba/WWW2020-O2MAC)]

### Graph Attention AE-based Community Detection

__Attributed graph clustering: A deep attentional embedding approach__. IJCAI 2019. _Wang et al._. [[Paper](https://www.ijcai.org/Proceedings/2019/0509.pdf)]

__Multi-view attribute graph convolution networks for clustering__. IJCAI 2020. _Cheng et al._. [[Paper](https://www.ijcai.org/Proceedings/2020/0411.pdf)]

__Deep multi-graph clustering via attentive cross-graph association__. WSDM 2020. _Luo et al._. [[Paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371806)] [[Code](https://github.com/flyingdoog/DMGC)]

### Variational AE-based Community Detection

__Stochastic blockmodels meet graph neural networks__. ICML 2019. _Mehta et al._. [[Paper](http://proceedings.mlr.press/v97/mehta19a/mehta19a.pdf)] [[Code](https://github.com/nikhil-dce/SBM-meet-GNN)]

__Variational graph embedding and clustering with laplacian eigenmaps__. IJCAI 2019. _Chen et al._. [[Paper](https://www.ijcai.org/Proceedings/2019/0297.pdf)]

__Learning community structure with variational autoencoder__. ICDM 2018. _Choong et al._. [[Paper](https://ieeexplore.ieee.org/document/8594831)]

__Effective decoding in graph auto-encoder using triadic closure__. AAAI 2020. _Shi et al._. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5437/5293)]

__Graph representation learning via ladder gamma variational autoencoders__. AAAI 2020. _Sarkar et al._. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6013/5869)]

__Adversarially regularized graph autoencoder for graph embedding__. IJCAI 2018. _Pan et al._. [[Paper](https://www.ijcai.org/Proceedings/2018/0362.pdf)] [[Code](https://github.com/Ruiqi-Hu/ARGA)]

__Optimizing variational graph autoencoder for community detection__. BigData 2019. _Choong et al._. [[Paper](https://ieeexplore.ieee.org/abstract/document/9006123)]


## Deep Nonnegative Matrix Factorization-based Community Detection 

__Deep autoencoder-like nonnegative matrix factorization for community detection__. CIKM 2018. _Ye et al._. [[Paper](https://dl.acm.org/doi/10.1145/3269206.3271697)] [[Code](https://github.com/benedekrozemberczki/DANMF)]

__A Non-negative Symmetric Encoder-Decoder Approach for Community Detection__. CIKM 2017. _Sun et al._. [[Paper](https://dl.acm.org/doi/abs/10.1145/3132847.3132902)]

__Community detection based on modularized deep nonnegative matrix factorization__. International Journal of Pattern Recognition and Artificial Intelligence 2020. _Huang et al._. [[Paper](https://www.worldscientific.com/doi/abs/10.1142/S0218001421590060)]

## Deep Sparse Filtering-based Community Detection
__Community discovery in networks with deep sparse filtering__. Pattern Recognition 2018. _Xie et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S003132031830116X)]


## Datasets
### Citation/Co-authorship Networks
- Citeseer, Cora, Pubmed https://linqs.soe.ucsc.edu/data
- DBLP http://snap.stanford.edu/data/com-DBLP.html
- Chemistry, Computer Science, Medicine, Engineering http://kddcup2016.azurewebsites.net/
### Online Social Networks
- Facebook http://snap.stanford.edu/data/ego-Facebook.html
- Epinions http://www.epinions.com/
- Youtube http://snap.stanford.edu/data/com-Youtube.html
- Last.fm https://www.last.fm/
- LiveJournal http://snap.stanford.edu/data/soc-LiveJournal1.html
- Gplus http://snap.stanford.edu/data/ego-Gplus.html
### Traditional Social Networks
- Cellphone Calls, http://www.cs.umd.edu/hcil/VASTchallenge08/
- Enron Mail, http://www.cs.cmu.edu/~enron/
- Friendship https://dl.acm.org/doi/10.1145/2501654.2501657
- Rados http://networkrepository.com/ia-radoslaw-email.php 
- Karate, Football,Dolphin http://www-personal.umich.edu/~mejn/netdata/
### Webpage Networks
- IMDb https://www.imdb.com/
- Wiki https://linqs.soe.ucsc.edu/data
### Product Co-purchasing Networks
- Amazon http://snap.stanford.edu/data/#amazon
### Other Networks
- Internet http://www-personal.umich.edu/~mejn/netdata/
- Java https://github.com/gephi/gephi/wiki/Datasets
- Hypertext http://www.sociopatterns.org/datasets
 

## Tools
- Gephi, https://gephi.org/
- Pajek, http://mrvar.fdv.uni-lj.si/pajek/
- LFR, https://www.santofortunato.net/resources

----------
**Disclaimer**

If you have any questions, please feel free to contact me.
Email: <u>fanzhen.liu@hdr.mq.edu.au</u>
