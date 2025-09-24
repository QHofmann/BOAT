import time
import copy
from typing import Dict, Any, Callable
from boat_jit.utils.op_utils import (
    copy_parameter_from_list,
    average_grad,
    manual_update,
)

import jittor as jit
from jittor.optim import Optimizer
import boat_jit.higher_jit as higher

importlib = __import__("importlib")
from boat_jit.operation_registry import get_registered_operation
from boat_jit.dynamic_ol import makes_functional_dynamical_system
from boat_jit.hyper_ol import makes_functional_hyper_operation

import matplotlib.pyplot as plt
import os
import json


def _load_loss_function(loss_config: Dict[str, Any]) -> Callable:
    module_name, func_name = loss_config["function"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return lambda *args, **kwargs: func(
        *args, **{**loss_config.get("params", {}), **kwargs}
    )


class Problem:
    """
    Jittor 版的 bi-level 优化 Problem，与 Torch 版在接口和逻辑上对齐。
    """

    def __init__(self, config: Dict[str, Any], loss_config: Dict[str, Any]):
        self._fo_gm = config["fo_gm"]
        self._dynamic_op = config["dynamic_op"]
        self._hyper_op = config["hyper_op"]
        self._ll_model = config["lower_level_model"]
        self._ul_model = config["upper_level_model"]
        self._ll_var = config["lower_level_var"]
        self._ul_var = config["upper_level_var"]
        self.boat_configs = config

        if config["dynamic_op"] is not None:
            if "GDA" in config["dynamic_op"]:
                assert (
                    loss_config.get("gda_loss", None) is not None
                ), "Set the 'gda_loss' in loss_config properly."
                self.boat_configs["gda_loss"] = _load_loss_function(
                    loss_config["gda_loss"]
                )

        self._lower_loop = config.get("lower_iters", 10)
        self._lower_opt = self.boat_configs["lower_level_opt"]
        self._upper_opt = self.boat_configs["upper_level_opt"]
        self._ll_loss = _load_loss_function(loss_config["lower_level_loss"])
        self._ul_loss = _load_loss_function(loss_config["upper_level_loss"])
        self._ll_solver = None
        self._ul_solver = None
        self._lower_init_opt = None
        self._fo_gm_solver = None
        self._log_results = []
        self._track_opt_traj = False
        self.loss_log_path = config["loss_log_path"]
        self.loss_history = []

    def build_ll_solver(self):
        if self.boat_configs["fo_gm"] is None:
            assert (self.boat_configs["dynamic_op"] is not None) and (
                self.boat_configs["hyper_op"] is not None
            ), "Set 'dynamic_op' and 'hyper_op' properly."

            self.check_status()

            # DM 需要辅助变量与优化器
            if "DM" in self._dynamic_op:
                self.boat_configs["DM"]["auxiliary_v"] = [
                    jit.zeros_like(param) for param in self._ll_var
                ]
                # 这些辅助变量需要参与更新
                for v in self.boat_configs["DM"]["auxiliary_v"]:
                    v.start_grad()

                self.boat_configs["DM"]["auxiliary_v_opt"] = jit.nn.SGD(
                    self.boat_configs["DM"]["auxiliary_v"],
                    lr=self.boat_configs["DM"]["auxiliary_v_lr"],
                )

            sorted_ops = sorted([op.upper() for op in self._dynamic_op])
            self._ll_solver = makes_functional_dynamical_system(
                custom_order=sorted_ops,
                ll_objective=self._ll_loss,
                ul_objective=self._ul_loss,
                ll_model=self._ll_model,
                ul_model=self._ul_model,
                lower_loop=self._lower_loop,
                solver_config=self.boat_configs,
            )

            # DI：为“动态初始化”准备一个学习率不同的下层优化器
            if "DI" in self.boat_configs["dynamic_op"]:
                opt_cls = type(self._upper_opt)
                di_lr = float(self.boat_configs["DI"]["lr"])
                new_groups = []
                for g in self._lower_opt.param_groups:
                    ng = {k: v for k, v in g.items() if k != "params"}
                    ng["params"] = g["params"]
                    ng["lr"] = di_lr
                    new_groups.append(ng)
                print(new_groups)
                # FIX
                self._lower_init_opt = opt_cls(new_groups, lr=di_lr)

        else:
            # 一阶方法（FOGM）
            self._fo_gm_solver = get_registered_operation(
                "%s" % self.boat_configs["fo_gm"]
            )(
                ll_objective=self._ll_loss,
                ul_objective=self._ul_loss,
                ll_model=self._ll_model,
                ul_model=self._ul_model,
                lower_loop=self._lower_loop,
                ll_var=self._ll_var,
                ul_var=self._ul_var,
                solver_config=self.boat_configs,
            )
        return self

    def build_ul_solver(self):
        if self.boat_configs["fo_gm"] is None:
            assert (
                self.boat_configs["hyper_op"] is not None
            ), "Choose hyper_op properly when not using FOGM."
            sorted_ops = sorted([op.upper() for op in self._hyper_op])
            self._ul_solver = makes_functional_hyper_operation(
                custom_order=sorted_ops,
                ul_objective=self._ul_loss,
                ll_objective=self._ll_loss,
                ll_model=self._ll_model,
                ul_model=self._ul_model,
                ll_var=self._ll_var,
                ul_var=self._ul_var,
                solver_config=self.boat_configs,
            )
        else:
            # FOGM 模式下通常不需要独立的 hyper_op
            assert (
                self.boat_configs["fo_gm"] is not None
            ), "FOGM is enabled, UL solver is handled by FOGM op."
        return self

    def run_iter(
        self,
        ll_feed_dict: Dict[str, jit.Var],
        ul_feed_dict: Dict[str, jit.Var],
        current_iter: int,
    ) -> tuple:
        if self.boat_configs["fo_gm"] is not None:
            start_time = time.perf_counter()

            # FIX: 对齐 Torch 版的 fogm_batch_input（支持批输入/元学习场景）
            if self.boat_configs.get("fogm_batch_input", False):
                for batch_ll_feed_dict, batch_ul_feed_dict in zip(
                    ll_feed_dict, ul_feed_dict
                ):
                    self._log_results.append(
                        self._fo_gm_solver.optimize(
                            batch_ll_feed_dict, batch_ul_feed_dict, current_iter
                        )
                    )
            else:
                self._log_results.append(
                    self._fo_gm_solver.optimize(ll_feed_dict, ul_feed_dict, current_iter)
                )

            run_time = time.perf_counter() - start_time

        else:
            run_time = 0.0
            if self.boat_configs["accumulate_grad"]:
                # 累积梯度：ll_feed_dict / ul_feed_dict 均为 list[dict]
                for batch_ll_feed_dict, batch_ul_feed_dict in zip(
                    ll_feed_dict, ul_feed_dict
                ):
                    with higher.innerloop_ctx(
                        self._ll_model,
                        self._lower_opt,
                        copy_initial_weights=False,
                        track_higher_grads=self._track_opt_traj,
                    ) as (auxiliary_model, auxiliary_opt):
                        forward_time = time.perf_counter()
                        dynamic_results = self._ll_solver.optimize(
                            ll_feed_dict=batch_ll_feed_dict,
                            ul_feed_dict=batch_ul_feed_dict,
                            auxiliary_model=auxiliary_model,
                            auxiliary_opt=auxiliary_opt,
                            current_iter=current_iter,
                        )
                        self._log_results.append(dynamic_results)
                        max_loss_iter = list(dynamic_results[-1].values())[-1]
                        forward_time = time.perf_counter() - forward_time

                        backward_time = time.perf_counter()
                        # FIX: 这里原来把 ul_feed_dict 误传成了 batch_ll_feed_dict，导致超层梯度用错数据
                        if "DM" not in self._dynamic_op:
                            self._log_results.append(
                                self._ul_solver.compute_gradients(
                                    ll_feed_dict=batch_ll_feed_dict,
                                    ul_feed_dict=batch_ul_feed_dict,  # FIX
                                    auxiliary_model=auxiliary_model,
                                    max_loss_iter=max_loss_iter,
                                )
                            )
                        else:
                            # DM 情况下与下方非累积分支一致：直接用 UL loss 反传
                            self._log_results.append(
                                self._ul_loss(
                                    batch_ul_feed_dict, self._ul_model, auxiliary_model
                                )
                            )
                        backward_time = time.perf_counter() - backward_time

                    run_time += forward_time + backward_time

                # 对 UL 参数做平均梯度（list 模式下 len(ll_feed_dict) 等于 batch 数）
                average_grad(self._ul_model, len(ll_feed_dict))

            else:
                with higher.innerloop_ctx(
                    self._ll_model,
                    self._lower_opt,
                    copy_initial_weights=False,
                    track_higher_grads=self._track_opt_traj,
                ) as (auxiliary_model, auxiliary_opt):
                    forward_time = time.perf_counter()
                    dynamic_results = self._ll_solver.optimize(
                        ll_feed_dict=ll_feed_dict,
                        ul_feed_dict=ul_feed_dict,
                        auxiliary_model=auxiliary_model,
                        auxiliary_opt=auxiliary_opt,
                        current_iter=current_iter,
                    )
                    self._log_results.append(dynamic_results)
                    max_loss_iter = list(dynamic_results[-1].values())[-1]
                    forward_time = time.perf_counter() - forward_time
                    print("forward_time", forward_time)

                    backward_time = time.perf_counter()
                    if "DM" not in self._dynamic_op:
                        self._log_results.append(
                            self._ul_solver.compute_gradients(
                                ll_feed_dict=ll_feed_dict,
                                ul_feed_dict=ul_feed_dict,
                                auxiliary_model=auxiliary_model,
                                max_loss_iter=max_loss_iter,
                            )
                        )
                    else:
                        # DM：直接用 UL loss 反传（与 Torch 版一致）
                        self._log_results.append(
                            self._ul_loss(ul_feed_dict, self._ul_model, auxiliary_model)
                        )
                    backward_time = time.perf_counter() - backward_time
                    print("backward_time", backward_time)

                    if self.boat_configs["copy_last_param"]:
                        copy_parameter_from_list(
                            self._ll_model, list(auxiliary_model.parameters(time=-1))
                        )

                # DI：用 _lower_init_opt 对 LL var 做一次“动态初始化”更新
                if "DI" in self.boat_configs["dynamic_op"]:
                    manual_update(
                        self._lower_init_opt, self._lower_opt.param_groups[0]["params"]
                    )

                run_time = forward_time + backward_time

        # 计算/记录损失 & 更新上层参数
        if isinstance(ll_feed_dict, list):
            ll_fd = ll_feed_dict[0]
            ul_fd = ul_feed_dict[0]
        else:
            ll_fd = ll_feed_dict
            ul_fd = ul_feed_dict

        if not self.boat_configs["return_grad"]:
            # 不返回梯度：直接用保存的 _custom_grad 更新上层参数
            manual_update(self._upper_opt, self._ul_var)
        else:
            # 返回 UL 的梯度列表（与 Torch 版 return_grad 行为一致）
            ll_loss = self._ll_loss(ll_fd, self._ul_model, self._ll_model)
            ul_loss = self._ul_loss(ul_fd, self._ul_model, self._ll_model)
            print(f"ll_loss: {ll_loss.item()}  ul_loss: {ul_loss.item()}")
            self.save_losses(current_iter=current_iter, ll_loss=ll_loss, ul_loss=ul_loss)
            return [var._custom_grad for var in list(self._ul_var)], run_time

        ll_loss = self._ll_loss(ll_fd, self._ul_model, self._ll_model)
        ul_loss = self._ul_loss(ul_fd, self._ul_model, self._ll_model)
        print(f"ll_loss: {ll_loss.item()}  ul_loss: {ul_loss.item()}")
        self.save_losses(current_iter=current_iter, ll_loss=ll_loss, ul_loss=ul_loss)
        return self._log_results, run_time

    def set_track_trajectory(self, track_traj=True):
        self._track_opt_traj = track_traj

    def check_status(self):
        if any(item in self._hyper_op for item in ["PTT", "IAD", "RAD"]):
            self.set_track_trajectory(True)
        if "DM" in self.boat_configs["dynamic_op"]:
            assert (self.boat_configs["hyper_op"] == ["RAD"]) or (
                self.boat_configs["hyper_op"] == ["CG"]
            ), "When 'DM' is chosen, set the 'truncate_iter' properly."
        if "RGT" in self.boat_configs["hyper_op"]:
            assert (
                self.boat_configs["RGT"]["truncate_iter"] > 0
            ), "When 'RGT' is chosen, set the 'truncate_iter' properly ."
        if self.boat_configs["accumulate_grad"]:
            assert (
                "IAD" in self.boat_configs["hyper_op"]
            ), "When using 'accumulate_grad', only 'IAD' based methods are supported."
        if self.boat_configs["GDA"]["alpha_init"] > 0.0:
            assert (
                0.0 < self.boat_configs["GDA"]["alpha_decay"] <= 1.0
            ), "Parameter 'alpha_decay' used in method BDA should be in the interval (0,1)."
        if "FD" in self._hyper_op:
            assert (
                self.boat_configs["RGT"]["truncate_iter"] == 0
            ), "One-stage method doesn't need trajectory truncation."

        def check_model_structure(base_model, meta_model):
            for param1, param2 in zip(base_model.parameters(), meta_model.parameters()):
                # jittor 参数没有 device 属性；只校验 shape / dtype
                if (param1.shape != param2.shape) or (param1.dtype != param2.dtype):
                    return False
            return True

        if "IAD" in self._hyper_op:
            assert check_model_structure(self._ll_model, self._ul_model), (
                "With IAD or FOA operation, 'upper_level_model' and 'lower_level_model' have the same structure, "
                "and 'lower_level_var' and 'upper_level_var' are the same group of variables."
            )
        assert (("DI" in self._dynamic_op) ^ ("IAD" in self._hyper_op)) or (
            ("DI" not in self._dynamic_op) and ("IAD" not in self._hyper_op)
        ), "Only one of the 'PTT' and 'RGT' methods could be chosen."
        assert (
            0.0 <= self.boat_configs["GDA"]["alpha_init"] <= 1.0
        ), "Parameter 'alpha' used in method BDA should be in the interval (0,1)."
        assert (
            self.boat_configs["RGT"]["truncate_iter"] < self.boat_configs["lower_iters"]
        ), "The value of 'truncate_iter' shouldn't be greater than 'lower_loop'."

    def plot_losses(self):
        iters = [x["iter"] for x in self.loss_history]
        ll_losses = [x["ll_loss"] for x in self.loss_history]
        ul_losses = [x["ul_loss"] for x in self.loss_history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(iters, ll_losses, label="Lower-level Loss", color="blue")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Lower-level Loss")
        axes[0].legend(loc="upper left")
        axes[0].grid(True)

        axes[1].plot(iters, ul_losses, label="Upper-level Loss", color="orange")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Upper-level Loss")
        axes[1].legend(loc="upper left")
        axes[1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(self.loss_log_path), "loss_curve.png")
        plt.savefig(save_path)
        plt.close()

    def save_losses(self, current_iter, ll_loss, ul_loss):
        self.loss_history.append(
            {
                "iter": current_iter,
                "ll_loss": float(ll_loss.item()),
                "ul_loss": float(ul_loss.item()),
            }
        )
        with open(self.loss_log_path, "w") as f:
            json.dump(self.loss_history, f)
