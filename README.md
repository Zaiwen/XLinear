## (AAAI26) XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs

This is the official implementation of **XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs**.

paper:

[![arXiv](https://img.shields.io/badge/arXiv-2601.09237-b31b1b.svg)](
https://doi.org/10.48550/arXiv.2601.09237)

If you find this work useful, please cite:

```bibtex

AAAI version
@article{Chen_Jin_Huang_Feng_2026,
title={XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs},
volume={40},
url={https://ojs.aaai.org/index.php/AAAI/article/view/39121},
DOI={10.1609/aaai.v40i24.39121},
number={24},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Chen, Xinyang and Jin, Huidong and Huang, Yu and Feng, Zaiwen},
year={2026},
month={Mar.},
pages={20325-20335} }

arxiv version
@article{chen2026xlinear,
  title={XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs},
  author={Chen, Xinyang and Jin, Huidong and Huang, Yu and Feng, Zaiwen},
  journal={arXiv preprint arXiv:2601.09237},
  year={2026}
}
```

### Introduction
---
We have designed an extremely simple and efficient time series model—XLinear—based on MLP and sigmoid to handle real-world forecasting tasks with exogenous inputs, bridging the gap between efficiency and accuracy in time series forecasting.

### Overall Arctictrue
---
XLinear consists solely of two sets of gating modules with identical structures, which are designed to filter out noisy features in the temporal and variable dimensions, enhance critical features, and strengthen temporal patterns within the time series, respectively. To mitigate information interference between different dimensions, we draw on the approach proposed in TimeXer for learning global representations of endogenous variables, thereby facilitating information integration across these two dimensions.
<div align='center'>
  <img width="800" height="500" alt="image" src="./fig/fig2.png" />
</div>

### Main Results
---
First, we conduct forecasting tasks with exogenous variables on 7 commonly used datasets. For this scenario, we designate the last variable as the endogenous variable, with the remaining variables serving as exogenous variables.
<div align='center'>
  <img width="800" height="600" alt="image" src="./fig/tab2.png" />
</div>

Second, we supplement 5 additional datasets with strong exogenous factor interference for comparative experiments. To comprehensively evaluate the model's performance in hydrological forecasting scenarios, we incorporate new metrics such as NSE, KGE, and MAPE to assess its effectiveness.
<div align='center'>
  <img width="800" height="600" alt="image" src="./fig/tab3.png" />
</div>

Furthermore, we investigate the performance of XLinear in multivariate forecasting scenarios.
<div align='center'>
  <img width="800" height="600" alt="image" src="./fig/tab4.png" />
</div>

### Model Analysis
---
#### Efficiency
In addition to achieving exceptional accuracy, XLinear maintains remarkably high efficiency, reaching a level comparable to that of DLinear and RLinear.
We investigated the efficiency of XLinear in both multivariate forecasting scenarios and univariate forecasting scenarios with exogenous inputs. Although DLinear and RLinear outperform XLinear by a marginal advantage in terms of efficiency, their predictive accuracy is considerably inferior. In contrast, compared with Transformer-based models that achieve higher accuracy, XLinear exhibits an approximate 30% improvement in training speed while consuming less GPU memory.
<div align="center">
  <div style="display: flex; gap: 10px; justify-content: center;">
    <img src="./fig/fig1.png" alt="Figure 1" width="320" height="220" style="object-fit: cover;"/>
    <img src="./fig/fig4.png" alt="Figure 2" width="320" height="220" style="object-fit: cover;"/>
  </div>
</div>

#### Long Lookback Window
Furthermore, we investigate the capability of XLinear to learn temporal patterns from longer lookback windows. 
<div align='center'>
  <img width="320" height="220" alt="image" src="./fig/fig7.png" />
</div>

Concurrently, we compare it with several lightweight and high-precision time series models in terms of variations in model resource consumption and running speed as the lookback window expands.
<div align='center'>
  <img width="500" height="320" alt="image" src="./fig/fig8.png" />
</div>


### Usage
---
1.Datasets can be obtained via the following links: 
[ETT](https://github.com/zhouhaoyi/ETDataset)
[Weather](https://www.bgc-jena.mpg.de/wetter/)
[Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
[Traffic](https://pems.dot.ca.gov/)
[Crop](https://www.kaggle.com/datasets/ajithdari/crop-yield-prediction)
[DO_409202](http://data.water.vic.gov.au/WMIS/)
[DO_425012](realtimedata.waternsw.com.au/water.stm)

2.Install Pytorch and other necessary dependencies.
```bash
pip install -r requirements.txt
```

3.All dataset scripts are centralized in the `script` folder. Execute the following startup commands in the main directory. Examples are as follows:
```bash
bash ./script/multi_forcasting/etth1.sh
```

### Concat
If you have any questions or concerns, please contact us at {Warren.Jin@csiro.au, Yhuang@mail.hzau.edu.cn, Zaiwen.Feng@mail.hzau.edu.cn} or submit an issue.
