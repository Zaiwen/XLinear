import torch
from torch import nn
from MLPAer import Model as MLPAer

class params_config():
    def __init__(self, dataset):
        if dataset == 'etth1':
            self.param = [96, 96, 512, 7, 512, 14, 0, 0, 0.7, 0]
        elif dataset == 'ettm1':
            self.param = [96, 96, 512, 7, 256, 7, 0.2, 0, 0.4, 0.2]
        elif dataset == 'weather':
            self.param = [96, 96, 2048, 21, 512, 48, 0.1, 0.1, 0.5, 0.1]
        elif dataset == 'electricity':
            self.param = [96, 96, 2048, 321, 512, 32, 0.2, 0.1, 0.3, 0.2]
        elif dataset == 'traffic':
            self.param = [96, 96, 2048, 862, 336, 64, 0.3, 0.1, 0.7, 0.3]

class MLPAer_Config():
    def __init__(self, param):
        self.seq_len = param[0]
        self.pred_len = param[1]
        self.d_model = param[2]
        self.enc_in = param[3]
        self.t_ff = param[4]
        self.c_ff = param[5]
        self.t_dropout = param[6]
        self.c_dropout = param[7]
        self.head_dropout = param[8]
        self.embed_dropout = param[9]

# the metric function
def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == "__main__":
    pa = params_config('traffic')
    config = MLPAer_Config(pa.param)
    model = MLPAer(config)
    test_params_flop(model, (96, 862))