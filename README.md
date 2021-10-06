# Compressed Image Restoration
- different codecs / codec information in bitstream
- different encoding parameters
- spatial/temporal artifacts
- generalization --- modulation --- spatial-wise - gumbel / global prior/local prior/auxiliary learning for qp, codecs predictions 
- some note: modulation: consider codec information, global prior, local prior, prototype (graph different codec relationships)
- learning degradation module
- self-supervised loss 

|  Title   | year  | Venue | code | Motivation | Solution| Key Technical Points|Thinking|
|  ----  | ----  | ---- | ---- | ---- | ----| ---- | ---- |
|[Deep Maximum a Posterior Estimator for Video Denoising](https://link.springer.com/content/pdf/10.1007/s11263-021-01510-7.pdf)  | 2021 | IJCV| |MAP-based framework |
|[BlockCNN: A Deep Network for Artifact Removal and Image Compression](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w50/Maleki_BlockCNN_A_Deep_CVPR_2018_paper.pdf)  | 2018 | CVPRW | |Pixel CNN |
|[BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment](https://arxiv.org/pdf/2104.13371v1.pdf)  | 2021 | CVPR| | |
|Adaptive Spatial-Temporal Fusion of Multi-Objective Networks for Compressed Video Perceptual Enhancement | 2021 | CVPRW| ||
|Boosting the Performance of Video Compression Artifact Reduction with Reference Frame Proposals and Frequency Domain Information | 2021 | CVPRW| ||
|[Deep Likelihood Network for Image Restoration with Multiple Degradation Levels](https://arxiv.org/pdf/1904.09105.pdf)  | 2021 | TIP| |MAP-based framework |
|[Toward Interactive Modulation for Photo-Realistic Image Restoration](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Cai_Toward_Interactive_Modulation_for_Photo-Realistic_Image_Restoration_CVPRW_2021_paper.pdf)  | 2021 | CVPR| | use encoding information to predict shift parameters for modulation in GAN-based framework|
|[Enhanced Separable Convolution Network for Lightweight JPEG Compression Artifacts Reduction](https://ieeexplore.ieee.org/document/9463776)  | 2021 | SPL| |  |
|[NTIRE 2021 Challenge on Quality Enhancement of Compressed Video:Methods and Results](https://arxiv.org/pdf/2104.10781.pdf)  | 2021 | CVPR| | competition |
|[Boosting the Performance of Video Compression Artifact Reduction with Reference Frame Proposals and Frequency Domain Information](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Xu_Boosting_the_Performance_of_Video_Compression_Artifact_Reduction_With_Reference_CVPRW_2021_paper.pdf)  | 2021 | CVPR| | competition |
|[Patch-Wise Spatial-Temporal Quality Enhancement for HEVC Compressed Video](https://ieeexplore.ieee.org/abstract/document/9477424)  | 2021 | TIP| |  |
|[MRS-Net+ for Enhancing Face Quality of Compressed Videos](https://ieeexplore.ieee.org/abstract/document/9509420)  | 2021 | TCSVT| |  |
|[Interlayer Restoration Deep Neural Network for Scalable High Efficiency Video Coding](https://ieeexplore.ieee.org/abstract/document/9478899)  | 2021 | TCSVT| |  |
|[Compressed Domain Deep Video Super-Resolution](https://ieeexplore.ieee.org/abstract/document/9509352)  | 2021 | TIP| |  |
|[A Model-Driven Deep Unfolding Method for JPEG Artifacts Removal](https://xueyangfu.github.io/paper/2021/TNNLS/paper.pdf)  | 2021 | TNNLS| |  |
|[Spatio-Temporal Deformable Convolution for Compressed Video Quality Enhancement](https://aaai.org/ojs/index.php/AAAI/article/view/6697/6551)|2020|AAAI| | |
|[Early Exit or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images](https://arxiv.org/pdf/2006.16581.pdf)|2020|ECCV| | |
|[Degradation Model Learning for Real-World Single Image Super-resolution](http://www4.comp.polyu.edu.hk/~cslzhang/paper/ACCV20-DML.pdf)|2020|ACCV| | |
|[Interactive Multi-Dimension Modulation with Dynamic Controllable Residual Learning for Image Restoration](https://arxiv.org/pdf/1912.05293.pdf)  | 2020 | ECCV| | weight generating, generalization|
|[Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700239.pdf)  | 2020 | ECCV| | weight generating, generalization|
|[When to use Convolutional Neural Networks for Inverse Problems](https://ci2cv.net/media/papers/nate_cvpr20.pdf)  | 2020 | CVPR| | |
|[Quantization Guided JPEG Artifact Correction](https://arxiv.org/abs/2004.09320)  | 2020 | ECCV| | |
|[JPEG Artifacts Removal via Compression Quality Ranker-Guided Networks](https://www.ijcai.org/proceedings/2020/0079.pdf)  | 2020 | IJCAI| |Existing CNN\-based methods only consider pixel\-level loss functions to guide network training. JPEG compression factor has not been fully utilized, and is also hard to use due to the non-differentiability.| They first learn a CNN\-based quality ranker to measure the compression degree, and then propose one quality related loss and one feature matching loss to guide de\-blocking and perceptual quality optimization. Noted that they use dilated convolutions to extract multi-scale features.| The CNN\-based quality ranker aims to prediction quality score of the input image, and is based on Siamese network. To train it, they use rank-based loss due to lacking ground-truth quality score. The quality related loss and feature matching loss use the final output and intermediate output to constraint the quality enhancement network training. | 1.The prediction quality score can be also used as input. 2. quality predictor feature can be used as an attention.|
|[Learning a Single Model With a Wide Range of Quality Factors for JPEG Image Artifacts Removal](https://arxiv.org/abs/2009.06912)  | 2020 | TIP| [link](https://github.com/VDIGPKU/QGCN) |These methods usually train a model based on one specific value or a small range of quality factors| quantization table as input for adapting to different qp, global pooling for global prior|utilization of the quantization tables as part of the training data, it has two branches in parallel, the restoration branch and the global branch｜
|[Multi-level Wavelet-Based Generative Adversarial Network for Perceptual Quality Enhancement of Compressed Video](https://arxiv.org/pdf/2008.00499.pdf)|2020|ECCV| | |
|[Compressed Image Restoration via Artifacts-Free PCA Basis Learning and Adaptive Sparse Modeling](https://ieeexplore.ieee.org/abstract/document/9121762)  | 2020 | TIP| |  |
|[Learning Local and Global Priors for JPEG Image Artifacts Removal](https://ieeexplore.ieee.org/document/9269390)  | 2020 | SPL| | global prior (global pooling), local prior (channel-wise attention), UNet, weak improvement |
|[Learning Continuous Image Representation with Local Implicit Image Function](https://arxiv.org/pdf/2012.09161.pdf)  | 2020 | CVPR| | |:star: :star:|
|[MFQE 2.0: A New Approach for Multi-frame Quality Enhancement on Compressed Video](https://arxiv.org/pdf/1902.09707.pdf)|2019|TPAMI| [link](https://github.com/RyanXingQL/MFQEv2.0)| |
|[Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Modulating_Image_Restoration_With_Continual_Levels_via_Adaptive_Feature_Modification_CVPR_2019_paper.pdf)|2019|CVPR| | |
|[CFSNet: Toward a Controllable Feature Space for Image Restoration](https://arxiv.org/pdf/1904.00634.pdf)  | 2019 | ICCV| | weight generating, generalization|
|[Deep Network Interpolation for Continuous Imagery Effect Transition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Deep_Network_Interpolation_for_Continuous_Imagery_Effect_Transition_CVPR_2019_paper.pdf)  | 2019 | CVPR| | weight generating, generalization|
|[Deep Non-Local Kalman Network for Video Compression Artifact Reduction](https://ieeexplore.ieee.org/abstract/document/8852849)|2019|TIP| | |
|[Non-Local ConvLSTM for Video Compression Artifact Reduction](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Non-Local_ConvLSTM_for_Video_Compression_Artifact_Reduction_ICCV_2019_paper.pdf)|2019|ICCV| | |
|[Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers](https://arxiv.org/pdf/1904.08118.pdf)| 2019 | CVPR| [link](https://github.com/hejingwenhejingwen/AdaFM) | AdaFM, feature modulation, generalization|:star::star::star:|
|[JPEG Artifacts Reduction via Deep Convolutional Sparse Coding](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_JPEG_Artifacts_Reduction_via_Deep_Convolutional_Sparse_Coding_ICCV_2019_paper.pdf)|2019|ICCV| | |:star::star:|
|[HEVC Compression Artifact Reduction with Generative Adversarial Networks](https://ieeexplore.ieee.org/document/8927915)|2019|WCSP| | |
|[A Comprehensive Benchmark for Single Image Compression Artifacts Reduction](https://arxiv.org/pdf/1909.03647.pdf)|2019|arxiv| | overview|
|[Multi-Frame Quality Enhancement for Compressed Video](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Multi-Frame_Quality_Enhancement_CVPR_2018_paper.pdf)  | 2018 | CVPR| | 1. use high-quality frame to enhance low-quality frame, thus propose a high quality frame detector 2. motion compensation, spatial transformer 3. quality enhancement network. Insight: 1. motion compensation might be inaccurate 2. warped feature may be not useful due to degradation.| |
|[Learning a Single Convolutional Super-Resolution Network for Multiple Degradations](https://arxiv.org/pdf/1712.06116.pdf)  | 2018 | CVPR| | MAP-based framework, take degradation information as input| |
|[MGANet: A Robust Model for Quality Enhancement of Compressed Video](https://arxiv.org/pdf/1811.09150.pdf)  | 2018 | arixv| | |
|[Learning a Single Convolutional Super-Resolution Network for Multiple Degradations](https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_SRMD.pdf)  | 2018 | CVPR| | |
|[Reduction of Video Compression Artifacts Based on Deep Temporal Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8502045)  | 2018 | Access| | |
|[Dmcnn: Dual-Domain Multi-Scale Convolutional Neural Network for Compression Artifacts Removal](http://39.96.165.147/Pub%20Files/2018/zhxs-icip18.pdf)  | 2018 | ICIP| | |
|[Deep Generative Adversarial Compression Artifact Removal](https://openaccess.thecvf.com/content_ICCV_2017/papers/Galteri_Deep_Generative_Adversarial_ICCV_2017_paper.pdf)  | 2017 | ICCV| | |
|[Decoder-side HEVC quality enhancement with scalable convolutional neural network](https://ieeexplore.ieee.org/document/8019299)  | 2017 | ICME| | |
|[D3: Deep Dual-Domain Based Fast Restoration of JPEG-Compressed Images](https://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_D3_Deep_Dual-Domain_CVPR_2016_paper.pdf)  | 2016 | CVPR| | |
|[Compression Artifacts Removal Using Convolutional Neural Networks](https://arxiv.org/pdf/1605.00366.pdf)  | 2016 | WSCG| | |
|[Compression Artifacts Reduction by a Deep Convolutional Network](https://arxiv.org/pdf/1504.06993.pdf)  | 2015 | ICCV| | |

# Related works
|  Title   | year  | Venue | code | keyword | Insightful|
|  ----  | ----  | ---- | ---- | ---- | ----|
|[Invertible Denoising Network: A Light Solution for Real Noise Removal](https://arxiv.org/abs/2104.10546)  | 2021 | CVPR| | |
|[Semi-Supervised Video Deraining with Dynamical Rain Generator](https://arxiv.org/pdf/2103.07939.pdf)  | 2021 | CVPR| | |
|[From Rain Generation to Rain Removal](https://arxiv.org/pdf/2008.03580.pdf)  | 2021 | CVPR| | |
|[Learning Spatially-Variant MAP Models for Non-blind Image Deblurring](https://openaccess.thecvf.com/content/CVPR2021/papers/Dong_Learning_Spatially-Variant_MAP_Models_for_Non-Blind_Image_Deblurring_CVPR_2021_paper.pdf)  | 2021 | CVPR| | |
|[Test-Time Fast Adaptation for Dynamic Scene Deblurring via Meta-Auxiliary Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chi_Test-Time_Fast_Adaptation_for_Dynamic_Scene_Deblurring_via_Meta-Auxiliary_Learning_CVPR_2021_paper.pdf)  | 2021 | CVPR| | |
|[Explore Image Deblurring via Encoded Blur Kernel Space](https://arxiv.org/pdf/2104.00317.pdf)  | 2021 | CVPR| | deblur |
|[Deep Generative Prior](https://zhuanlan.zhihu.com/p/165050802)  | 2020 | ECCV| |  |
|[Unified Dynamic Convolutional Network for Super-Resolution with Variational Degradations](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Unified_Dynamic_Convolutional_Network_for_Super-Resolution_With_Variational_Degradations_CVPR_2020_paper.pdf)  | 2020 | CVPR| |  |
|[Neural Blind Deconvolution Using Deep Priors](https://arxiv.org/pdf/1908.02197.pdf)  | 2020 | CVPR| | MAP-based framework, two neural networks respectively for modeling image and blur kernel, the structure of NN and self-training method and regularization are very important.   |:star::star::star:|
|[Deep Image Prior](https://arxiv.org/pdf/1711.10925.pdf)  | 2018 | CVPR| |  |:star::star::star::star:|
|[On-Demand Learning for Deep Image Restoration](https://arxiv.org/pdf/1612.01380.pdf)  | 2017 | ICCV| |  |
|[Confidence Measure Guided Single Image De-raining](https://arxiv.org/pdf/1909.04207.pdf)  | 2019 | arxiv| | deraining |
|[Pixel-Adaptive Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Su_Pixel-Adaptive_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 2019 | CVPR| | content-aware convolution |
|[Dynamic Filter Networks](https://proceedings.neurips.cc/paper/2016/file/8bf1211fd4b7b94528899de0a43b9fb3-Paper.pdf)| 2017 | NIPS| |  |
|[DIP sharing](https://zhuanlan.zhihu.com/p/369445239)|  | zhihu| |  |
|[“Double-DIP” :Unsupervised Image Decomposition via Coupled Deep-Image-Priors](https://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/resources/DoubleDIP.pdf)| 2019 | CVPR| |  |


# Prior based for deraining
- sparse prior total variation

|  Title   | year  | Venue | code | keyword | Insightful|
|  ----  | ----  | ---- | ---- | ---- | ----|
|[Multi-Decoding Deraining Network and Quasi-Sparsity Based Training](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Multi-Decoding_Deraining_Network_and_Quasi-Sparsity_Based_Training_CVPR_2021_paper.pdf)  | 2021 | CVPR| | |
|[Joint Bi-layer Optimization for Single-image Rain Streak Removal](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.pdf)  | 2017 | ICCV| | |
|[Joint Convolutional Analysis and Synthesis Sparse Representation for Single Image Layer Separation](https://openaccess.thecvf.com/content_ICCV_2017/papers/Gu_Joint_Convolutional_Analysis_ICCV_2017_paper.pdf)  | 2017 | ICCV| | |
|[Rain Streak Removal Using Layer Priors](https://yu-li.github.io/paper/li_cvpr16_rain.pdf)  | 2016 | CVPR| | |
|[FastDeRain: A Novel Video Rain Streak Removal Method Using Directional Gradient Priors](https://yu-li.github.io/paper/li_cvpr16_rain.pdf)  | 2016 | CVPR| | |

# attention 
|  Title   | year  | Venue | code | keyword | Insightful|
|  ----  | ----  | ---- | ---- | ---- | ----|
| Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining| 2020 | CVPR| | |
