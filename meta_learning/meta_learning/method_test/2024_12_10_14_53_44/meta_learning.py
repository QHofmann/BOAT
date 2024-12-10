import torch
import numpy as np
import torch.nn.functional as F
import boat
from torch import nn
from torch.nn import functional as F
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
import math
import higher
from tqdm import tqdm
# print(torch.cuda.is_available())
def get_cnn_omniglot(hidden_size, n_classes):
    def conv_layer(ic, oc, ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True # When this is true is called the "transductive setting"
                           )
        )

    net =  nn.Sequential(
        conv_layer(1, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size, n_classes)
    )

    initialize(net)
    return net

def initialize(net):
    # initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            # m.bias.data = torch.ones(m.bias.data.size())
            # m.weight.data.zero_()
            m.bias.data.zero_()
    return net

device = torch.device("cpu")
dataset = omniglot("C:/Users/ASUS/Documents/GitHub/BOAT/data/omniglot", ways=5, shots=1, test_shots=15, meta_train=True,download=True)
test_dataset = omniglot("C:/Users/ASUS/Documents/GitHub/BOAT/data/omniglot", ways=5, shots=1, test_shots=15, meta_test=True,download=True)

meta_model = get_cnn_omniglot(64, 5)

initialize(meta_model)
kwargs = {'num_workers': 1, 'pin_memory': True}
batch_size=4
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=batch_size, **kwargs)

inner_opt = torch.optim.SGD(lr=0.1, params=meta_model.parameters())
outer_opt = torch.optim.Adam(meta_model.parameters(),lr=0.01)
y_lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=outer_opt, T_max=80000,
                                                            eta_min=0.001)
import json
with open("C:/Users/ASUS/Documents/GitHub/BOAT/configs/boat_config_ml.json", "r") as f:
    boat_config = json.load(f)

with open("C:/Users/ASUS/Documents/GitHub/BOAT/configs/loss_config_ml.json", "r") as f:
    loss_config = json.load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Data HyperCleaner')

    parser.add_argument('--dynamic_method', type=str, default='',help='omniglot or miniimagenet or tieredImagenet')
    parser.add_argument('--hyper_method', type=str, default='',help='convnet for 4 convs or resnet for Residual blocks')
    parser.add_argument('--fo_', type=str, default='',help='convnet for 4 convs or resnet for Residual blocks')
    args = parser.parse_args()
    boat_config['lower_level_model'] = meta_model
    boat_config['upper_level_model'] = meta_model
    boat_config['lower_level_var'] = meta_model.parameters()
    boat_config['upper_level_var'] = meta_model.parameters()
    b_optimizer = boat.Problem(boat_config,loss_config)
    b_optimizer.build_ll_solver(inner_opt)
    b_optimizer.build_ul_solver(outer_opt)

    with tqdm(dataloader, total=10, desc="Meta Training Phase") as pbar:
        for meta_iter, batch in enumerate(pbar):
            ul_feed_dict = [{"data": batch["test"][0][k].to(device), "target": batch["test"][1][k].to(device)} for k in range(batch_size)]
            ll_feed_dict = [{"data": batch["train"][0][k].to(device), "target": batch["train"][1][k].to(device)} for k in range(batch_size)]
            loss, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=meta_iter)
            y_lr_schedular.step()
            print('validation loss:', loss)
            if meta_iter>=5:
                break
if __name__ == '__main__':
    main()
