import os
import torch
import boat_torch as boat
from torchmeta.toy.helpers import sinusoid
from torchmeta.utils.data import BatchMetaDataLoader

from tqdm import tqdm
import sys
from examples.meta_learning.meta_learning import get_sinuoid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

batch_size = 4
kwargs = {"num_workers": 1, "pin_memory": True}
device = torch.device("cpu")
dataset = sinusoid(shots=10, test_shots=100, seed=0)
meta_model = get_sinuoid()
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
test_dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
inner_opt = torch.optim.SGD(lr=0.1, params=meta_model.parameters())
outer_opt = torch.optim.Adam(meta_model.parameters(), lr=0.01)
y_lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=outer_opt, T_max=80000, eta_min=0.001
)
import os
import json

base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)
with open(os.path.join(parent_folder, "configs/boat_config_ml.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(parent_folder, "configs/loss_config_ml.json"), "r") as f:
    loss_config = json.load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data HyperCleaner")

    parser.add_argument(
        "--gradient_mapping",
        type=str,
        default="",
        help="omniglot or miniimagenet or tieredImagenet",
    )
    parser.add_argument(
        "--numerical_approximation",
        type=str,
        default="",
        help="convnet for 4 convs or resnet for Residual blocks",
    )
    parser.add_argument(
        "--fo_go",
        type=str,
        default="",
        help="convnet for 4 convs or resnet for Residual blocks",
    )
    args = parser.parse_args()

    gradient_mapping = args.gradient_mapping.split(",") if args.gradient_mapping else None
    numerical_approximation = args.numerical_approximation.split(",") if args.numerical_approximation else None
    print(args.gradient_mapping)
    print(args.numerical_approximation)
    boat_config["gradient_mapping_op"] = gradient_mapping
    boat_config["numerical_approximation_op"] = numerical_approximation
    boat_config["lower_level_model"] = meta_model
    boat_config["upper_level_model"] = meta_model
    boat_config["lower_level_var"] = list(meta_model.parameters())
    boat_config["upper_level_var"] = list(meta_model.parameters())
    boat_config["lower_level_opt"] = inner_opt
    boat_config["upper_level_opt"] = outer_opt
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()

    with tqdm(dataloader, total=1, desc="Meta Training Phase") as pbar:
        for meta_iter, batch in enumerate(pbar):
            ul_feed_dict = [
                {
                    "data": batch["test"][0][k].float().to(device),
                    "target": batch["test"][1][k].float().to(device),
                }
                for k in range(batch_size)
            ]
            ll_feed_dict = [
                {
                    "data": batch["train"][0][k].float().to(device),
                    "target": batch["train"][1][k].float().to(device),
                }
                for k in range(batch_size)
            ]
            # print(ll_feed_dict[0]['data'].shape,ll_feed_dict[0]['target'].shape)
            loss, run_time = b_optimizer.run_iter(
                ll_feed_dict, ul_feed_dict, current_iter=meta_iter
            )
            y_lr_schedular.step()
            print("validation loss:", loss[-1][-1])
            if meta_iter >= 1:
                break


if __name__ == "__main__":
    main()
