# l2_regularization.py  —— pytest 友好版（小数据/小维度/少循环 + DM/GDA 兜底 + CPU 默认）
import argparse, numpy as np, sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import jittor as jit
import boat_jit as boat
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.decomposition import TruncatedSVD





# -------------------- 工具函数：解包/转 numpy & 评估 --------------------
def _unwrap(x):
    while isinstance(x, (tuple, list)) and len(x) == 1:
        x = x[0]
    return x

def to_numpy(x):
    x = _unwrap(x)
    if isinstance(x, jit.Var):
        return x.numpy()
    return np.asarray(x)

def evaluate(W, testset):
    with jit.no_grad():
        test_x, test_y = testset
        W      = _unwrap(W)
        test_x = _unwrap(test_x)
        test_y = _unwrap(test_y)
        logits = _unwrap(test_x @ W)

        if not isinstance(test_y, jit.Var):
            test_y = jit.array(to_numpy(test_y), dtype=jit.int64)
        loss = jit.nn.cross_entropy_loss(logits, test_y).item()

        pred_np = to_numpy(_unwrap(logits.argmax(dim=-1)))
        true_np = to_numpy(test_y)
        acc = (pred_np == true_np).mean()
    return loss, acc

# -------------------- 小数据 & 小维度 --------------------
def get_data(args):
    train_x, train_y = fetch_20newsgroups_vectorized(
        subset="train", return_X_y=True, data_home=args.data_path, download_if_missing=True
    )
    test_x,  test_y  = fetch_20newsgroups_vectorized(
        subset="test",  return_X_y=True, data_home=args.data_path, download_if_missing=True
    )

    # 取一小部分样本，CI 更稳
    max_train = min(args.max_train, train_x.shape[0])
    max_test  = min(args.max_test,  test_x.shape[0])
    train_x   = train_x[:max_train]; train_y = train_y[:max_train]
    test_x    = test_x[:max_test];   test_y  = test_y[:max_test]

    # 划分
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, stratify=train_y, test_size=0.5, random_state=args.seed
    )
    test_x,  teval_x, test_y, teval_y = train_test_split(
        test_x,  test_y,  stratify=test_y,  test_size=0.5, random_state=args.seed
    )

    # SVD 降维（小到 256/512）
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
    train_x = svd.fit_transform(train_x); val_x = svd.transform(val_x)
    test_x  = svd.transform(test_x);     teval_x = svd.transform(teval_x)

    # 转 jittor Var
    train_x = jit.array(train_x, dtype=jit.float32)
    val_x   = jit.array(val_x,   dtype=jit.float32)
    test_x  = jit.array(test_x,  dtype=jit.float32)
    teval_x = jit.array(teval_x, dtype=jit.float32)
    train_y = jit.array(train_y, dtype=jit.int64)
    val_y   = jit.array(val_y,   dtype=jit.int64)
    test_y  = jit.array(test_y,  dtype=jit.int64)
    teval_y = jit.array(teval_y, dtype=jit.int64)

    print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)

# -------------------- 配置 --------------------
base_folder = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_folder, "configs_jit/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)
with open(os.path.join(base_folder, "configs_jit/loss_config_l2.json"), "r") as f:
    loss_config = json.load(f)

def main():

    parser = argparse.ArgumentParser()
    # —— 小配置，适合 pytest / CI ——
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--svd_dim", type=int, default=256)
    parser.add_argument("--max_train", type=int, default=500)
    parser.add_argument("--max_test", type=int, default=500)
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dm_op", type=str, default="DI,NGD")
    parser.add_argument("--na_op", type=str, default="RAD,RGT,PTT")
    parser.add_argument("--fo_op", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed); jit.set_global_seed(args.seed)

    # ---- CI/pytest 默认禁用 GPU，避免 jittor cuda 侧崩溃 ----
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        try:
            jit.flags.use_cuda = 0
        except Exception:
            pass

    # ---- 兜底必须的配置键（DM/GDA/RGT/日志）----
    os.makedirs(args.data_path, exist_ok=True)
    boat_config.setdefault("loss_log_path", os.path.join(args.data_path, "loss_log.json"))
    boat_config.setdefault("accumulate_grad", False)
    boat_config.setdefault("return_grad", False)
    boat_config.setdefault("copy_last_param", False)
    boat_config.setdefault("lower_iters", 1)

    # DM
    boat_config.setdefault("DM", {})
    boat_config["DM"].setdefault("auxiliary_v_lr", 0.01)

    # GDA
    boat_config.setdefault("GDA", {})
    boat_config["GDA"].setdefault("alpha_init", 0.0)  # 允许 0
    boat_config["GDA"].setdefault("alpha_decay", 0.5) # (0,1]

    # RGT
    boat_config.setdefault("RGT", {})
    boat_config["RGT"].setdefault("truncate_iter", 1)

    # 读取数据
    trainset, valset, testset, tevalset = get_data(args)

    # 可选：保存个小数据，方便复用
    jit.save((trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg_small.pkl"))
    print(f"[info] saved tiny data to {args.data_path}/l2reg_small.pkl")

    # -------------------- 简单线性模型 --------------------
    class UpperModel(jit.Module):
        def __init__(self, n_feats):
            super().__init__()
            self.x = jit.array(np.zeros((n_feats,), dtype=np.float32)); self.x.start_grad()
        def execute(self): return self.x

    class LowerModel(jit.Module):
        def __init__(self, n_feats, num_classes):
            super().__init__()
            self.W = jit.array(np.zeros((n_feats, num_classes), dtype=np.float32)); self.W.start_grad()
        def execute(self): return self.W

    n_feats = trainset[0].shape[1]
    n_cls   = int(trainset[1].max().item()) + 1
    upper_model = UpperModel(n_feats)
    lower_model = LowerModel(n_feats, n_cls)
    upper_opt = jit.nn.Adam([upper_model.x], lr=0.01)
    lower_opt = jit.nn.SGD([lower_model.W], lr=0.01)

    # -------------------- 组装 BOAT 配置 --------------------
    dm_op = args.dm_op.split(",") if args.dm_op else None
    na_op   = args.na_op.split(",")   if args.na_op   else None
    if na_op is not None and ("RGT" in na_op):
        boat_config["RGT"]["truncate_iter"] = 1

    boat_config["dm_op"]       = dm_op
    boat_config["na_op"]         = na_op
    boat_config["fo_op"]            = args.fo_op
    boat_config["lower_level_model"]= lower_model
    boat_config["upper_level_model"]= upper_model
    boat_config["lower_level_opt"]  = lower_opt
    boat_config["upper_level_opt"]  = upper_opt
    boat_config["lower_level_var"]  = list(lower_model.parameters())
    boat_config["upper_level_var"]  = list(upper_model.parameters())

    # FOGM 场景：只传 fo_op（pytest 的 fo_op 单测就是这种）
    # （由 boat.Problem 内部的 FOGM 分支处理）
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver().build_ul_solver()

    # 数据喂法：与之前版本一致
    ul_feed_dict = {"data": trainset[0], "target": trainset[1]}
    ll_feed_dict = {"data": valset[0],   "target": valset[1]}

    # 迭代次数：对齐你 torch 版最小逻辑
    if boat_config["dm_op"] is not None:
        if ("DM" in boat_config["dm_op"]) and ("GDA" in boat_config["dm_op"]):
            iterations = 3
        else:
            iterations = 2
            # 需要上层梯度时可打开（按需）：
            # b_optimizer.boat_configs["return_grad"] = True
    else:
        iterations = 2
    iterations = min(iterations, args.iterations)

    for it in range(iterations):
        # DM+GDA 的 strategy 切换（如有 DM）
        if boat_config["dm_op"] is not None:
            if ("DM" in boat_config["dm_op"]) and ("GDA" in boat_config["dm_op"]):
                b_optimizer._ll_solver.gradient_instances[-1].strategy = "s" + str((it % 3) + 1)
            elif ("DM" in boat_config["dm_op"]) and ("GDA" not in boat_config["dm_op"]):
                b_optimizer._ll_solver.gradient_instances[-1].strategy = "s1"

        _, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=it)

        # 评估：只用下层权重矩阵 W
        te_loss, te_acc = evaluate(lower_model(), testset)
        tv_loss, tv_acc = evaluate(lower_model(), tevalset)
        print(f"[info] iter {it:2d} test loss {te_loss:.4f} acc {te_acc:.4f}  teval loss {tv_loss:.4f} acc {tv_acc:.4f}  time {run_time:.2f}")

if __name__ == "__main__":
    main()
