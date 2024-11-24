<div align="center"> Official repository of <br> &nbsp; </div>


<div align="center">
    <img width="25%" src="https://raw.githubusercontent.com/jankolf/assets/main/QUD/qud-logo.png">
</div>
<div align="center">[ /kjuːt/ ] &nbsp; </div>

# <div align="center"> QUD: Unsupervised Knowledge Distillation<br>for Deep Face Recognition</div>

<div align="center", padding="30px">
  <span style="display:block; height: 20em;">&emsp;</span>
  <p><b>Jan Niklas Kolf</b><sup> 1,2</sup> &emsp; <b>Naser Damer</b><sup> 1,2</sup> &emsp; <b>Fadi Boutros</b><sup> 1</sup></p>
  <p><sup>1 </sup>Fraunhofer IGD &emsp; <sup>2 </sup>Technische Universität Darmstadt</p>
  <span style="display:block; height: 50em;">&emsp;</span>
  <p>Accepted at BMVC 2024</p>
  <span style="display:block; height: 50em;">&emsp;</span>
</div>

<div align="center">
        <a href="https://share.jankolf.de/s/3xxiaEKN4qCBHpC"><img src="https://github.com/jankolf/assets/blob/main/badges/download-paper-arxiv-c1.svg?raw=true"></a>
        &emsp;
        <a href="https://share.jankolf.de/s/MmXSzqpsoNW8t3A"><img src="https://github.com/jankolf/assets/blob/main/badges/download-data-c1.svg?raw=true"></a>
</div>


## <div align="center"> Overview 🔎 </div>
<div align="center">
    <img width="100%" src="https://raw.githubusercontent.com/jankolf/assets/main/QUD/qud_overview_github.svg">
</div>

Overview of the proposed unsupervised KD method QUD. Using contrastive loss, student S is trained so that the distance of its feature f = S(x) to teacher T’s positive feature f<sup>+</sup> = T(x) of the same input sample x is smaller than the distance between f and a set of negative features f<sup>-</sup> ∈ __f__<sup>-</sup> stored in a queue. The queue is filled with features of T from previous iterations. After each iteration, the current features of T are enqueued and an equal amount of the oldest features are dequeued from the queue. Only S’s parameters are updated.


## <div align="center"> Abstract 🤏 </div>
We present in this paper an unsupervised knowledge distillation (KD) approach, namely QUD, for face recognition. The proposed QUD approach utilizes a queue of features within a contrastive learning setup to guide the student model to learn a feature representation similar to its counterpart obtained from the teacher and dissimilar from the ones that are stored in a queue. This queue is updated by pushing a batch of feature representations obtained from the teacher into the queue and dequeuing the oldest ones from the queue in each training iteration. We additionally incorporate a temperature into the contrastive loss to control how sensitive contrastive learning is to samples considered negative in the queue. The proposed unsupervised QUD approach does not require accessing the same dataset used to train the teacher model or even for the data to have identity labels. The effectiveness of the proposed approach is demonstrated through several sensitivity studies on different teacher architectures and using different datasets for student training in the KD framework. Additionally, the achieved results on mainstream benchmarks by our unsupervised QUD are compared to state-of-the-art (SOTA), achieving very competitive performances and even outperforming SOTA on several benchmarks.

## <div align="center"> Citation ✒ </div>
If you found this work helpful for your research, please cite the article with the following bibtex entry:
```
@inproceedings{DBLP:conf/bmvc/KolfQUD,
  author       = {Jan Niklas Kolf and
                  Naser Damer and
                  Fadi Boutros},
  title        = {{QUD:} Unsupervised Knowledge Distillation for Deep Face Recognition},
  booktitle    = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow,
                  UK, November 25-28, 2024},
  publisher    = {{BMVA} Press},
  year         = {2024},
}
```

## License ##

This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

Copyright (c) 2024 Fraunhofer Institute for Computer Graphics Research IGD, Darmstadt.
