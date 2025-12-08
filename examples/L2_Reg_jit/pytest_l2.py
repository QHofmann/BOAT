import pytest
import subprocess

# 动态方法组合
gm_oplist = (
    ["GDA", "NGD", "DI"],
    ["DI", "NGD", "GDA"],
    ["NGD"],
    ["DI", "NGD"],
    ["GDA", "NGD"],
)

# 超参数方法组合
na_oplist = (
    ["CG"],
    ["CG", "PTT"],
    ["RAD"],
    ["RAD", "PTT"],
    ["RAD", "RGT"],
    ["PTT", "RAD", "RGT"],
    ["FD"],
    ["FD", "PTT"],
    ["NS"],
    ["NS", "PTT"],
    ["IGA"],
    ["IGA", "PTT"],
)

# 带 DM 的组合
gm_op_dm = (
    ["DM", "NGD"],
    ["DM", "GDA", "NGD"],
)
na_op_dm = (
    ["RAD"],
    ["CG"],
)

# fo_op 方法
fogm_method = (
    ["VSO"],
    ["VFO"],
    ["MESO"],
    ["PGDO"],
)

# 脚本路径 (注意用 r'' 原始字符串，避免 \ 转义问题)
#SCRIPT = r"/public/home/panjibao/project/jit/examples/L2_Reg_jit/l2_regularization_org.py"
SCRIPT = r"/public/home/panjibao/project/jit/jit_127/BOAT/examples/L2_Reg_jit/l2_regularization.py"





@pytest.mark.parametrize("fogm_method", fogm_method)
def test_fogm_method(fogm_method):
    command = [
        "python",
        SCRIPT,
        "--fo_op",
        fogm_method[0],  # 取 tuple 里的单个字符串
    ]
    print(f"Running test with fo_op={fogm_method}")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Test failed for fo_op={fogm_method}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )





@pytest.mark.parametrize(
    "gm_op, na_op",
    [
        (gm_op, na_op)
        for gm_op in gm_op_dm
        for na_op in na_op_dm
    ],
)
def test_combination_dynamic_na_op_dm(gm_op, na_op):
    command = [
        "python",
        SCRIPT,
        "--gm_op",
        ",".join(gm_op),
        "--na_op",
        ",".join(na_op),
    ]
    print(f"Running test with gm_op={gm_op}, na_op={na_op}")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Test failed for gm_op={gm_op}, na_op={na_op}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )






@pytest.mark.parametrize(
    "gm_op, na_op",
    [
        (gm_op, na_op)
        for gm_op in gm_oplist
        for na_op in na_oplist
    ],
)
def test_combination_dynamic_na_op(gm_op, na_op):
    command = [
        "python",
        SCRIPT,
        "--gm_op",
        ",".join(gm_op),
        "--na_op",
        ",".join(na_op),
    ]
    print(f"Running test with gm_op={gm_op}, na_op={na_op}")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Test failed for gm_op={gm_op}, na_op={na_op}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

