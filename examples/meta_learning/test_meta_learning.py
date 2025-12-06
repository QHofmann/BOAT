import os
import time
import shutil
import subprocess
import pytest
import platform

# 假设 meta_learning.py 已经被复制到正确的位置
gradient_mappinglist = (["NGD"], ["NGD", "GDA"])
numerical_approximationlist = (
    ["IAD"],
    ["IAD", "PTT"],
    ["CG", "IAD"],
    ["CG", "IAD", "PTT"],
    ["NS", "IAD"],
    ["NS", "IAD", "PTT"],
    ["FOA", "IAD"],
    ["FOA", "IAD", "PTT"],
)
fo_ol_method = (["VSO"], ["VFO"], ["MESO"], ["PGDO"])

# 获取当前时间
t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "meta_learning/method_test"  # 使用相对路径

# 获取当前脚本所在的目录（相对路径）
base_folder = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
folder = os.path.join(base_folder, args, t0)  # 构建相对路径

# 创建文件夹
if not os.path.exists(folder):
    os.makedirs(folder)

# 将 Python 文件复制到目标文件夹
ganfolder = os.path.join(folder, "meta_learning.py")
shutil.copyfile(os.path.join(base_folder, "meta_learning.py"), ganfolder)

# 创建一个临时的 shell 脚本（Windows 下是 .bat 文件）
script_extension = ".bat" if platform.system() == "Windows" else ".sh"
script_file = os.path.join(folder, "set" + script_extension)

# 创建批处理或 shell 脚本
with open(script_file, "w") as f:
    k = 0
    for gradient_mapping in gradient_mappinglist:
        for numerical_approximation in numerical_approximationlist:
            k += 1
            f.write(
                f'python /home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py --gradient_mapping {",".join(gradient_mapping)} --numerical_approximation {",".join(numerical_approximation)} \n'
            )

# 如果是 Ubuntu 系统, 使得脚本具有执行权限
if platform.system() != "Windows":
    os.chmod(script_file, 0o775)  # 给 sh 文件执行权限


# 使用 pytest.mark.parametrize 进行参数化
@pytest.mark.parametrize(
    "gradient_mapping, numerical_approximation",
    [
        (gradient_mapping, numerical_approximation)
        for gradient_mapping in gradient_mappinglist
        for numerical_approximation in numerical_approximationlist
    ],
)
def test_combination_dynamic_numerical_approximation(gradient_mapping, numerical_approximation):
    # 构建命令
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py",
        "--gradient_mapping",
        ",".join(gradient_mapping),
        "--numerical_approximation",
        ",".join(numerical_approximation),
    ]
    print(
        f"Running test with gradient_mapping={gradient_mapping} and numerical_approximation={numerical_approximation}"
    )

    result = subprocess.run(command, capture_output=True, text=True)

    # 确保命令执行成功
    assert (
        result.returncode == 0
    ), f"Test failed for gradient_mapping={gradient_mapping} and numerical_approximation={numerical_approximation}. Error: {result.stderr}"

@pytest.mark.parametrize("fo_ol_method", fo_ol_method)
def test_fo_ol_method(fo_ol_method):
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/meta_learning/meta_learning.py",
        "--fo_op",
        fo_ol_method[0],
    ]
    print(f"Running test with fo_op={fo_ol_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for fo_op={fo_ol_method}. Error: {result.stderr}"
