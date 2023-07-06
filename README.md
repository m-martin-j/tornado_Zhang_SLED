# The Tornado Framework

![Language](https://img.shields.io/badge/language-Python-blue.svg)
![Stars](https://img.shields.io/github/stars/alipsgh/tornado?color=r)
![Repo Size](https://img.shields.io/github/repo-size/alipsgh/tornado?color=tomato)

**This fork is created using the code provided by Zhang et al.: https://github.com/shuxiangzhang/drift-detection**
```bibtex
@inproceedings{Zhang.2020,
 author = {Zhang, Shuxiang and {Jung Huang}, David Tse and Dobbie, Gillian and Koh, Yun Sing},
 title = {SLED: Semi-supervised Locally-weighted Ensemble Detector},
 pages = {1838--1841},
 publisher = {IEEE},
 isbn = {978-1-7281-2903-7},
 booktitle = {2020 IEEE 36th International Conference on Data Engineering},
 year = {2020},
 address = {Piscataway, NJ},
 doi = {10.1109/ICDE48307.2020.00183},
 file = {7fe4be67-85cc-47de-8375-54aaed37c4d6:C\:\\Users\\trat\\AppData\\Local\\Swiss Academic Software\\Citavi 6\\ProjectCache\\l2u09mheovwvhdo4ur83y1sr1fky7wxbrpl3bc482hcib683s\\Citavi Attachments\\7fe4be67-85cc-47de-8375-54aaed37c4d6.pdf:pdf}
}
```

<p align="center">
  <img src="/img/tornado.png" width="25%"/>
</p>

**Tornado** is a framework for data stream mining, implemented in Python. The framework includes various incremental/online learning algorithms as well as concept drift detection methods.

You must have Python 3.5 or above (either 32-bit or 64-bit) on your system to run the framework without any error. Note that the **numpy**, **scipy**, **matplotlib**, and **pympler** packages are used in the Tornado implementations. You may use the `pip` command in order to install these packages, for example:

```bash
pip install numpy
```

Although you can use an installer from https://www.python.org/downloads/ to install Python on your system, I highly recommend **Anaconda**, one of the Python distributions, since it includes the **numpy**, **scipy**, and **mathplotlib** packages by default. You may download one of the Anaconda's installers from https://www.anaconda.com/download/. Please note that, you still need to install the **pympler** package for Anaconda. For that, run the following command in a command prompt or a terminal:

```bash
conda install -c conda-forge pympler
```

Once you have all the packages installed, you may run the framework.

Three sample codes are prepared to show how you can use the framework. Those files are:
* **_github_prequential_test.py_** - This file lets you evaluate an adaptive algorithm, i.e. a pair of a learner and a drift detector, prequentially. In this example, Naive Bayes is the learner and Fast Hoeffding Drift Detection Method (FHDDM) is the detector. You find lists of incremental learners in `tornado/classifier/` and drift detectors in `tornado/drift_detection/`. The outputs in the created project directory are similar to:

<p align="center">
  <img src="/tutorial_img/pr/nb_fhddm.100.png" width="50%"/><br />
  <img src="/tutorial_img/pr/nb_fhddm.100.er.png" width="40%"/>
</p>

* **_github_prequential_multi_test.py_** - This file lets you run multiple adaptive algorithms together against a data stream. While algorithms are learning from instances of a data stream, the framework tells you which adaptive algorithm is optimal by considering _classification_, _adaptation_, and _resource consumption_ measures. The outputs in the created project directory are similar to:

<p align="center">
  <img src="/tutorial_img/multi/sine1_multi_score.png" width="80%"/><br />
  <img src="/tutorial_img/multi/sine1_multi_sine1_cr.png" width="75%"/>
</p>

* **_github_generate_stream.py_** - The file helps you use the Tornado framework for generating synthetic data streams containing concept drifts. You find a list of stream generators in `tornado/streams/generators/`.

### Citation

Please kindly cite the following papers, or thesis, if you plan to use Tornado or any of its components:

1. Pesaranghader, Ali. "__A Reservoir of Adaptive Algorithms for Online Learning from Evolving Data Streams__", Ph.D. Dissertation, Université d'Ottawa/University of Ottawa, 2018. <br />
DOI: http://dx.doi.org/10.20381/ruor-22444
2. Pesaranghader, Ali, et al. "__Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams__", *Machine Learning Journal*, 2018. <br />
Pre-print available at: https://arxiv.org/abs/1709.02457, DOI: https://doi.org/10.1007/s10994-018-5719-z
3. Pesaranghader, Ali, et al. "__A framework for classification in data streams using multi-strategy learning__", *International Conference on Discovery Science*, 2016. <br />
Pre-print available at: http://iwera.ir/~ali/papers/ds2016.pdf, DOI: https://doi.org/10.1007/978-3-319-46307-0_22

<br/>
<br/>

<sub>Ali Pesaranghader © 2020++ | MIT License</sub>
