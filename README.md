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

----------
## Survey
__A Comprehensive Survey on Community Detection with Deep Learning__. **28 Pages**, arXiv 2021. _Xing Su, Shan Xue, Fanzhen Liu, Jia Wu, Jian Yang, Chuan Zhou, Wenbin Hu, Cecile Paris, Surya Nepal, Di Jin, Quan Z. Sheng, Philip S. Yu_. [[Paper](https://arxiv.org/pdf/2105.12584.pdf)] [[AMiner科技](https://www.aminer.cn/research_report/60da8c5f30e4d5752f50e7af)]

Link: https://arxiv.org/abs/2105.12584

    @article{su2021survey,
    	author = {Su, Xing and Xue, Shan and Liu, Fanzhen and Wu, Jia and Yang, Jian and 
		Zhou, Chuan and Hu, Wenbin and Paris, Cecile and Nepal, Surya and Jin, Di and 
		Sheng, Quan Z. and Yu, Philip S.},
		journal = {arXiv preprint arXiv:2105.12584},
		title = {A Comprehensive Survey on Community Detection with Deep Learning},
		year = {2021},
		url = {https://arxiv.org/abs/2105.12584}
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
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
| Community detection in node-attributed social networks: A survey | Computer Science Review | 2020 | _Petr Chunaev_ | [[Paper](https://www.sciencedirect.com/science/article/pii/S1574013720303865)] |
| Community detection in networks: A multidisciplinary review | Journal of Network and Computer Applications | 2018|  _Javed et al._ | [[Paper](https://www.sciencedirect.com/science/article/pii/S1084804518300560)] |
| Community discovery in dynamic networks: A Survey | ACM Computing Surveys | 2018 | _Giulio Rossetti and Remy Cazabet_ | [[Paper](https://dl.acm.org/doi/10.1145/3172867)] |
| Evolutionary computation for community detection in networks: A review | IEEE TEVC | 2018 | _Clara Pizzuti_ | [[Paper](https://ieeexplore.ieee.org/document/8004509)] |
| Metrics for community analysis: A survey | ACM Computing Surveys | 2017 | _Chakraborty et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3091106)] |
| Network community detection: A review and visual survey | Preprint | 2017 | _Bisma S. Khan and Muaz A. Niazi_ | [[Paper](https://arxiv.org/abs/1708.00977)] |
| Community detection in networks: A user guide | Physics Reports | 2016 | _Santo Fortunato and Darko Hric_ | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0370157316302964)] |
| Community detection in social networks | WIREs Data Mining Knowledge Discovery | 2016 | _Punam Bedi and Chhavi Sharma_ | [[Paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/widm.1178)]|

----------
## Convolutional Networks-based Community Detection
### CNN-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
|Edge classification based on convolutional neural networks for community detection in complex network | Physica A | 2020 | _Cai et al._ | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378437120304271)] |
|A deep learning based community detection approach | SAC | 2019 | _Giancarlo Sperlí_ | [[Paper](https://doi.org/10.1145/3297280.3297574)] |
|Deep community detection in topologically incomplete networks | Physica A | 2017 | _Xin et al._ | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378437116308342)] |

### GCN-based Community Detection 
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
|Adaptive graph encoder for attributed graph embedding | KDD | 2020 | _Cui et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3394486.3403140)][[Code](https://github.com/thunlp/AGE)] |
|CommDGI: Community detection oriented deep graph infomax | CIKM | 2020 | _Zhang et al._ | [[Paper](https://dl.acm.org/doi/abs/10.1145/3340531.3412042)] | 
|Going deep: Graph convolutional ladder-shape networks | AAAI | 2020 | _Hu et al._ | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5673/5529)] |
|Independence promoted graph disentangled networks | AAAI | 2020 | _Liu et al._ | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5929/5785)] |
|Supervised community detection with line graph neural networks | ICLR | 2019 | _Chen et al._ | [[Paper](https://openreview.net/pdf?id=H1g0Z3A9Fm)][[Code](https://github.com/zhengdao-chen/GNN4CD)] |
|Graph convolutional networks meet markov random fields: Semi-supervised community detection in attribute networks | AAAI | 2019 | _Jin et al._ | [[Paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3780/3658)] |
|Overlapping community detection with graph neural networks | DLG Workshop, KDD | 2019 | _Oleksandr Shchur and Stephan Günnemann_ | [[Paper](https://deep-learning-graphs.bitbucket.io/dlg-kdd19/accepted_papers/DLG_2019_paper_3.pdf)][[Code](https://github.com/shchur/overlapping-community-detection)] |
|Attributed graph clustering via adaptive graph convolution | IJCAI | 2019 | _Zhang et al._ | [[Paper](https://www.ijcai.org/Proceedings/2019/0601.pdf)][[Code](https://github.com/karenlatong/AGC-master)] |
|CayleyNets: Graph convolutional neural networks with complex rational spectral filters | IEEE Transactions on Signal Processing | 2019 |  _Levie et al._ | [[Paper](https://ieeexplore.ieee.org/document/8521593)][[Code](https://github.com/amoliu/CayleyNet)] |

----------
## Graph Attention Network-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | -- | ---- | ---- | 
|HDMI: High-order deep multiplex infomax | WWW | 2021 | _Jing et al._ | [[Paper](https://dl.acm.org/doi/fullHtml/10.1145/3442381.3449971)][[Code](https://github.com/baoyujing/HDMI)] |
|Self-supervised heterogeneous graph neural network with co-contrastive learning | KDD | 2021 | _Wang et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3447548.3467415)][[Code](https://github.com/liun-online/HeCo)] |
|Unsupervised attributed multiplex network embedding | AAAI | 2020 | _Park et al._ | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5985)][[Code](https://github.com/pcy1302/DMGI)] |
|MAGNN: Metapath aggregated graph neural network for heterogeneous graph embedding | WWW | 2020 | _Fu et al._ | [[Paper](https://dl.acm.org/doi/abs/10.1145/3366423.3380297)] [[Code](https://github.com/cynricfu/MAGNN)] |

----------
## Graph Adversarial Network-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|SEAL: Learning heuristics for community detection with generative adversarial networks | KDD | 2020 | _Zhang et al._ | [[Paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403154)][[Code](https://github.com/yzhang1918/kdd2020seal)] |
|Multi-class imbalanced graph convolutional network learning | IJCAI | 2020 | _Shi et al._ | [[Paper](https://www.ijcai.org/proceedings/2020/398)] |
|JANE: Jointly adversarial network embedding | IJCAI | 2020| _Yang et al._ | [[Paper](https://www.ijcai.org/Proceedings/2020/0192.pdf)] |
|ProGAN: Network embedding via proximity generative adversarial network | KDD | 2019 | _Gao et al._ | [[Paper](https://dl.acm.org/doi/pdf/10.1145/3292500.3330866)] |
|CommunityGAN: Community detection with generative adversarial nets | WWW | 2019 | _Jia et al._ | [[Paper](https://dl.acm.org/doi/abs/10.1145/3308558.3313564)][[Code](https://github.com/SamJia/CommunityGAN)] |

----------
## Autoencoder-based Community Detection
### Stacked AE-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|An evolutionary autoencoder for dynamic community detection | Science China Information Sciences | 2020 | _Wang et al._ | [[Paper](https://link.springer.com/article/10.1007/s11432-020-2827-9)] |
|High-performance community detection in social networks using a deep transitive autoencoder | Information Sciences | 2019 | _Xie et al._ | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025519303251)] |
|Integrative network embedding via deep joint reconstruction | IJCAI | 2018 | _Hongchang Gao and Heng Huang_ | [[Paper](https://www.ijcai.org/Proceedings/2018/0473.pdf)] |
|Deep attributed network embedding | IJCAI | 2018 | _Gao et al._ | [[Paper](https://www.ijcai.org/Proceedings/2018/0467.pdf)][[Code](https://github.com/gaoghc/DANE)] |
|Deep network embedding for graph representation learning in signed networks | IEEE TCYB | 2018 | _Xiao Sheng and Fu-Lai Chung_ | [[Paper](https://ieeexplore.ieee.org/document/8486671)][[Code](https://github.com/shenxiaocam/Deep-network-embedding-for-graph-representation-learning-in-signed-networks)] |
| BL-MNE: Emerging heterogeneous social network embedding through broad learning with aligned autoencoder | ICDM | 2017 | _Zhang et al._ | [[Paper](https://doi.org/10.1109/ICDM.2017.70)][[Code](http://www.ifmlab.org/files/code/Aligned-Autoencoder.zip)] |
|Modularity based community detection with deep learning | IJCAI | 2016 | _Yang et al._ | [[Paper](https://www.ijcai.org/Proceedings/16/Papers/321.pdf)][[Code](http://yangliang.github.io/code/DC.zip)] |

### Sparse AE-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|Stacked autoencoder-based community detection method via an ensemble clustering framework | Information Sciences | 2020 | _Xu et al._ | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S002002552030270X)] |
|DFuzzy: A deep learning-based fuzzy clustering model for large graphs | Knowledge and Information Systems | 2018 | _Vandana Bhatia and Rinkle Rani_ | [[Paper](https://link.springer.com/article/10.1007/s10115-018-1156-3)] |
|Learning deep representations for graph clustering | AAAI | 2014 | _Tian et al._ | [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8527/8571)][[Code](https://github.com/quinngroup/deep-representations-clustering)] |

### Denoising AE-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|MGAE: Marginalized graph autoencoder for graph clustering | CIKM | 2017 | _Wang et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3132847.3132967)][[Code](https://github.com/FakeTibbers/MGAE)] |
|Graph clustering with dynamic embedding | Preprint | 2017 | _Yang et al._ | [[Paper](https://arxiv.org/abs/1712.08249)] | 
|Deep neural networks for learning graph representations | AAAI | 2016 | _Cao et al._ | [[Paper](https://dl.acm.org/doi/10.5555/3015812.3015982)] |

### Graph Convolutional AE-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|Community-centric graph convolutional network for unsupervised community detection | IJCAI | 2020 | _He et al._ | [[Paper](https://www.ijcai.org/Proceedings/2020/0486.pdf)] |
|Structural deep clustering network |  WWW | 2020 | _Bo et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3366423.3380214)][[Code](https://github.com/bdy9527/SDCN)] |
|One2Multi graph autoencoder for multi-view graph clustering | WWW | 2020 | _Fan et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3366423.3380079)][[Code](https://github.com/googlebaba/WWW2020-O2MAC)] |

### Graph Attention AE-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|Multi-view attribute graph convolution networks for clustering | IJCAI | 2020 | _Cheng et al._ | [[Paper](https://www.ijcai.org/Proceedings/2020/0411.pdf)] |
|Deep multi-graph clustering via attentive cross-graph association | WSDM | 2020 | _Luo et al._ | [[Paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371806)][[Code](https://github.com/flyingdoog/DMGC)] |
|Attributed graph clustering: A deep attentional embedding approach | IJCAI | 2019 | _Wang et al._ | [[Paper](https://www.ijcai.org/Proceedings/2019/0509.pdf)] |

### Variational AE-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|Effective decoding in graph auto-encoder using triadic closure | AAAI | 2020 | _Shi et al._ | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5437/5293)] |
|Graph representation learning via ladder gamma variational autoencoders | AAAI | 2020 | _Sarkar et al._ | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6013/5869)] |
|Stochastic blockmodels meet graph neural networks | ICML | 2019 | _Mehta et al._ | [[Paper](http://proceedings.mlr.press/v97/mehta19a/mehta19a.pdf)][[Code](https://github.com/nikhil-dce/SBM-meet-GNN)] |
|Variational graph embedding and clustering with laplacian eigenmaps | IJCAI | 2019 | _Chen et al._ | [[Paper](https://www.ijcai.org/Proceedings/2019/0297.pdf)] |
|Optimizing variational graph autoencoder for community detection | BigData | 2019 | _Choong et al._ | [[Paper](https://ieeexplore.ieee.org/abstract/document/9006123)] |
|Learning community structure with variational autoencoder | ICDM | 2018 | _Choong et al._ | [[Paper](https://ieeexplore.ieee.org/document/8594831)] |
|Adversarially regularized graph autoencoder for graph embedding | IJCAI | 2018 | _Pan et al._ | [[Paper](https://www.ijcai.org/Proceedings/2018/0362.pdf)][[Code](https://github.com/Ruiqi-Hu/ARGA)]| 

----------
## Deep Nonnegative Matrix Factorization-based Community Detection 
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|Community detection based on modularized deep nonnegative matrix factorization | International Journal of Pattern Recognition and Artificial Intelligence | 2020 | _Huang et al._ | [[Paper](https://www.worldscientific.com/doi/abs/10.1142/S0218001421590060)] |
|Deep autoencoder-like nonnegative matrix factorization for community detection | CIKM | 2018 | _Ye et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3269206.3271697)][[Code](https://github.com/benedekrozemberczki/DANMF)] |
|A non-negative symmetric encoder-decoder approach for community detection | CIKM | 2017 | _Sun et al._ | [[Paper](https://dl.acm.org/doi/abs/10.1145/3132847.3132902)] |

----------
## Deep Sparse Filtering-based Community Detection
| Paper Title | Venue | Year | Authors | Materials | 
| ---- | ---- | ---- | ---- | ---- | 
|Community discovery in networks with deep sparse filtering | Pattern Recognition | 2018 | _Xie et al._ | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S003132031830116X)] |

----------
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
- Cellphone Calls http://www.cs.umd.edu/hcil/VASTchallenge08/
- Enron Mail http://www.cs.cmu.edu/~enron/
- Friendship https://dl.acm.org/doi/10.1145/2501654.2501657
- Rados http://networkrepository.com/ia-radoslaw-email.php 
- Karate, Football, Dolphin http://www-personal.umich.edu/~mejn/netdata/
### Webpage Networks
- IMDb https://www.imdb.com/
- Wiki https://linqs.soe.ucsc.edu/data
### Product Co-purchasing Networks
- Amazon http://snap.stanford.edu/data/#amazon
### Other Networks
- Internet http://www-personal.umich.edu/~mejn/netdata/
- Java https://github.com/gephi/gephi/wiki/Datasets
- Hypertext http://www.sociopatterns.org/datasets
 
 ----------
## Tools
- Gephi https://gephi.org/
- Pajek http://mrvar.fdv.uni-lj.si/pajek/
- LFR https://www.santofortunato.net/resources

----------
**Disclaimer**

If you have any questions, please feel free to contact us.
Emails: <u>fanzhen.liu@hdr.mq.edu.au</u>, <u>xing.su2@hdr.mq.edu.au</u>
