# XLinear

### Introduction
---
This paper aims to bridge the gap between efficiency and accuracy in time series forecasting, particularly in scenarios involving exogenous inputs. A feature-filtering gating module composed of MLP and sigmoid is constructed for this purpose. Furthermore, we leverage global tokens extracted from endogenous sequences to filter valid information from exogenous input sequences. The proposed method achieves state-of-the-art performance on 7 benchmark datasets and 5 real-world datasets with external inputs; meanwhile, compared with mainstream Transformer-based models, it exhibits a 30% improvement in running speed.
<div align="center">
  <img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/2463bce3-9667-4755-a868-5e2bcb59ac31" />
</div>

### Overall Arctictrue
---
XLinear comprises merely two sets of gating modules with identical structures, which are employed to capture temporal dimension features of endogenous variables and effective information in exogenous variables respectively. To prevent information interference between different dimensions, we draw on the approach in TimeXer for learning global representations for endogenous variables to bridge the information of these two dimensions.
<div align='center'>
  <img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/1b9b74a3-55bf-49d5-ac02-94365b3778cf" />
</div>

