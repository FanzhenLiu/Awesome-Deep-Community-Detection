# Awesome Deep Community Detection
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
  - [Datasets](#datasets)
  - [Tools](#tools)


## Survey
__Deep Learning for Community Detection: Progress, Challenges and Opportunities__. IJCAI 2020. _Fanzhen Liu, Shan Xue, Jia Wu, Chuan Zhou, Wenbin Hu, Cecile Paris, Surya Nepal, Jian Yang, Philip S. Yu_. [[Paper](https://www.ijcai.org/Proceedings/2020/0693.pdf)] [[AI科技评论](https://cloud.tencent.com/developer/article/1632305)] [[AI科技评论](https://cloud.tencent.com/developer/article/1632305)]


Link: https://www.ijcai.org/Proceedings/2020/693

    @inproceedings{ijcai2020-693,
    	author = {Liu, Fanzhen and Xue, Shan and Wu, Jia and Zhou, Chuan and Hu, Wenbin and 
		Paris, Cecile and Nepal, Surya and Yang, Jian and Yu, Philip S},
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


## Deep Neural Network-based Community Detection
### CNN-based Approaches
__A deep learning based community detection approach__. SAC 2019. _Giancarlo Sperlí_. [[Paper](https://doi.org/10.1145/3297280.3297574)]

__Deep community detection in topologically incomplete networks__. Physica A 2017. _Xin et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378437116308342)]

### Auto-encoder-based Approaches
__Graph representation learning via ladder gamma variational autoencoders__. AAAI 2020. _Sarkar et al._. [[Paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-SarkarA.4681.pdf)]

__Semi-implicit graph variational auto-encoders__. NIPS 2019. _Hasanzadeh et al._. [[Paper](https://papers.nips.cc/paper/9255-semi-implicit-graph-variational-auto-encoders.pdf)] [[Code](https://github.com/sigvae/SIGraphVAE)]

__Attributed graph clustering: A deep attentional embedding approach__. IJCAI 2019. _Wang et al._. [[Paper](https://www.ijcai.org/Proceedings/2019/0509.pdf)]

__Network-specific variational auto-encoder for embedding in attribute networks__. IJCAI 2019. _Chen et al._. [[Paper](https://www.ijcai.org/Proceedings/2019/0370.pdf)]

__Variational graph embedding and clustering with laplacian eigenmaps__. IJCAI 2019. _Chen et al_. [[Paper](https://www.ijcai.org/Proceedings/2019/0297.pdf)]

__Learning community structure with variational autoencoder__. ICDM 2018. _Choong et al._. [[Paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-SarkarA.4681.pdf)]

__Adversarially regularized graph autoencoder for graph embedding__. IJCAI 2018. _Pan et al._. [[Paper](https://www.ijcai.org/Proceedings/2018/0362.pdf)] [[Code](https://github.com/Ruiqi-Hu/ARGA)]

__Deep attributed network embedding__. IJCAI 2018. _Chen et al._. [[Paper](https://www.ijcai.org/Proceedings/2018/0467.pdf)] [[Code](https://github.com/gaoghc/DANE)]

__Integrative network embedding via deep joint reconstruction__. IJCAI 2018. _Hongchang Gao and Heng Huang_. [[Paper](https://www.ijcai.org/Proceedings/2018/0473.pdf)]

__Deep network embedding for graph representation learning in signed networks__. IEEE TCYB 2018. _Xiao Sheng and Fu-Lai Chung_. [[Paper](https://ieeexplore.ieee.org/document/8486671)] [[Code](https://github.com/shenxiaocam/Deep-network-embedding-for-graph-representation-learning-in-signed-networks)]

__Incorporating network structure with node contents for community detection on large networks using deep learning__. Neurocomputing 2018. _Cao et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231218300985)]	

__Autoencoder based community detection with adaptive integration of network topology and node contents__. KSEM 2018. _Cao et al._. [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-99247-1_16)]

__DFuzzy: A deep learning-based fuzzy clustering model for large graphs__. Knowledge and Information Systems 2018. _Vandana Bhatia and Rinkle Rani_. [[Paper](https://link.springer.com/article/10.1007/s10115-018-1156-3)]

__BL-MNE: Emerging heterogeneous social network embedding through broad learning with aligned autoencoder__. ICDM 2017. _Zhang et al._. [[Paper](https://doi.org/10.1109/ICDM.2017.70)] [[Code](http://www.ifmlab.org/files/code/Aligned-Autoencoder.zip)]

__Modularity based community detection with deep learning__. IJCAI 2016. _Yang et al._. [[Paper](https://www.ijcai.org/Proceedings/16/Papers/321.pdf)] [[Code](http://yangliang.github.io/code/DC.zip)]

### GAN-based Approaches
__JANE: Jointly adversarial network embedding__. IJCAI 2020. _Yang et al._. [[Paper](https://www.ijcai.org/Proceedings/2020/0192.pdf)]

__CommunityGAN: Community detection with generative adversarial nets__. WWW 2019. _Jia et al._. [[Paper](https://dl.acm.org/doi/abs/10.1145/3308558.3313564)] [[Code](https://github.com/SamJia/CommunityGAN)]

__Learning graph representation with generative adversarial nets__. IEEE TKDE 2019. _Wang et al._. [[Paper](https://ieeexplore.ieee.org/document/8941296)]

__GraphGAN: Graph representation learning with generative adversarial nets__. AAAI 2018. _Wang et al._. [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16611/15969)] [[Code](https://github.com/hwwang55/GraphGAN)]

## Deep Graph Embedding-based Community Detection
### Deep NMF-based Approaches
__Deep autoencoder-like nonnegative matrix factorization for community detection__. CIKM 2018. _Ye et al._. [[Paper](https://dl.acm.org/doi/10.1145/3269206.3271697)] [[Code](https://github.com/benedekrozemberczki/DANMF)]

### Deep SF-based Approaches
__Community discovery in networks with deep sparse filtering__. Pattern Recognition 2018. _Xie et al._. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S003132031830116X)]

### Community Embedding-based Approaches
__vGraph: A generative model for joint community detection and node representation learning.__. NIPS 2019. _Sun et al._. [[Paper](https://papers.nips.cc/paper/8342-vgraph-a-generative-model-for-joint-community-detection-and-node-representation-learning.pdf)] [[Code](https://github.com/fanyun-sun/vGraph)]

__A unified framework for community detection and network representation learning__. IEEE TKDE 2019. _Tu et al._. [[Paper](https://ieeexplore.ieee.org/document/8403293)] [[Code](http://nlp.csai.tsinghua.edu.cn/~tcc/datasets/simplified_CNRL.zip)]

__Embedding both finite and infinite communities on graphs__. IEEE Computational Intelligence Magazine 2019. _Cavallari et al._. [[Paper](https://ieeexplore.ieee.org/document/8764640)]

__Cosine: Community-preserving social network embedding from information
diffusion cascades__. AAAI 2018. _Zhang et al._. [[Paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16364/159824)]

__Learning community embedding with community detection and node embedding on graphs__. CIKM 2017. _Cavallari et al._. [[Paper](https://dl.acm.org/doi/10.1145/3132847.3132925)] [[Code](https://github.com/vwz/ComE)]

## Graph Neural Network-based Community Detection
__Community-centric graph convolutional network for unsupervised community detection__. IJCAI 2020. _He et al._. [[Paper](https://www.ijcai.org/Proceedings/2020/0486.pdf)]

__Diffusion improves graph learning__. NIPS 2019. _Klicpera et al._. [[Paper](https://papers.nips.cc/paper/9490-diffusion-improves-graph-learning.pdf)] [[Code](https://github.com/klicperajo/gdc)]

__End to end learning and optimization on graphs__. NIPS 2019. _Wilder et al._. [[Paper](https://papers.nips.cc/paper/8715-end-to-end-learning-and-optimization-on-graphs.pdf)] [[Code](https://github.com/bwilder0/clusternet)]

__Attributed graph clustering: A deep attentional embedding approach__. IJCAI 2019. _Wang et al._. [[Paper](https://www.ijcai.org/Proceedings/2019/0509.pdf)]

__Supervised community detection with line graph neural networks__. ICLR 2019. _Chen et al._. [[Paper](https://openreview.net/pdf?id=H1g0Z3A9Fm)] [[Code](https://github.com/zhengdao-chen/GNN4CD)]

__Overlapping community detection with graph neural networks__. Deep Learning on Graphs Workshop, SIGKDD 2019. _Oleksandr Shchur and Stephan Günnemann_. [[Paper](https://deep-learning-graphs.bitbucket.io/dlg-kdd19/accepted_papers/DLG_2019_paper_3.pdf)] [[Code](https://github.com/shchur/overlapping-community-detection)]

__Graph convolutional networks meet markov random fields: Semi-supervised community detection in attribute networks__. AAAI 2019. _Jin et al._. [[Paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3780/3658)]

__Adversarially regularized graph autoencoder for graph embedding__. IJCAI 2018. _Pan et al._. [[Paper](https://www.ijcai.org/Proceedings/2018/0362.pdf)] [[Code](https://github.com/Ruiqi-Hu/ARGA)]


## Datasets
- MEJN, http://www-personal.umich.edu/~mejn/netdata/
- SNAP, http://snap.stanford.edu/data/index.html
- Cellphone Calls, http://www.cs.umd.edu/hcil/VASTchallenge08/
- Enron Mail, http://www.cs.cmu.edu/~enron/
- Friendship https://dl.acm.org/doi/10.1145/2501654.2501657

## Tools
- Gephi, https://gephi.org/
- Pajek, http://mrvar.fdv.uni-lj.si/pajek/

----------
**Disclaimer**

If you have any questions, please feel free to contact me.
Email: <u>fanzhen.liu@hdr.mq.edu.au</u>
