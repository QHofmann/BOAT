
# BOAT --- Task-Agnostic Operation Toolbox for Gradient-based Bilevel Optimization
[![PyPI version](https://badge.fury.io/py/boml.svg)](https://badge.fury.io/py/boml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/callous-youth/BOAT/workflow.yml)
[![codecov](https://codecov.io/github/callous-youth/BOAT/graph/badge.svg?token=0MKAOQ9KL3)](https://codecov.io/github/callous-youth/BOAT)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/callous-youth/BOAT)
[![pages-build-deployment](https://github.com/callous-youth/BOAT/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/callous-youth/BOAT/actions/workflows/pages/pages-build-deployment)
![GitHub top language](https://img.shields.io/github/languages/top/callous-youth/BOAT)
![GitHub language count](https://img.shields.io/github/languages/count/callous-youth/BOAT)
![Python version](https://img.shields.io/badge/python-3.8-blue)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

**BOAT** is a task-agnostic, gradient-based **Bi-Level Optimization (BLO)** Python library that abstracts BLO into modular and flexible components, enabling efficient modeling of hierarchical and nested learning problems. It supports a wide spectrum of optimization settings, including first- and second-order methods, nested or non-nested formulations, with or without theoretical guarantees. This repository provides the **Jittor-based** implementation (boat-jit), leveraging Jittor’s JIT compilation and efficient CUDA/cuDNN backends to accelerate large-scale gradient-based BLO experiments.


To enhance flexibility and efficiency, BOAT incorporates the **Gradient Mapping Operation LIBRARY (GM-OL)** and the **Numerical Approximation Operation LIBRARY (NA-OL)**, alongside a collection of state-of-the-art first-order optimization strategies. BOAT also provides multiple implementation versions:
- **[PyTorch-based](https://github.com/callous-youth/BOAT)**: An efficient and widely-used version.
- **[Jittor-based](https://github.com/callous-youth/BOAT/tree/boat_jit)**: An accelerated version for high-performance tasks.
- **[MindSpore-based](https://github.com/callous-youth/BOAT/tree/boat_ms)**: Incorporating the latest first-order optimization strategies to support emerging application scenarios.


BOAT is designed to offer robust computational support for a broad spectrum of BLO research and applications, enabling innovation and efficiency in machine learning and computer vision.


## 🔑  **Key Features**
- **Gradient Mapping Operation LIBRARY (GM-OL)**: Incorporates 4 advanced dynamic system construction operations, enabling users to flexibly tailor optimization trajectories for BLO tasks.
- **Hyper-Gradient Operation Library (H-OL)**: Provides 9 refined operations for hyper-gradient computation, significantly enhancing the precision and efficiency of gradient-based BLO methods.
- **First-Order Operation Library (FO-Ol)**: Integrates 4 state-of-the-art first-order methods, enabling fast prototyping and validation of new BLO algorithms. With modularized design, BOAT allows flexible combinations of multiple upper-level and lower-level operators, resulting in over **63+16** (nearly 80) algorithmic combinations, offering unparalleled adaptability.
- **Modularized Design for Customization**: Empowers users to flexibly combine dynamic and hyper-gradient operations while customizing the specific forms of problems, parameters, and optimizer choices, enabling seamless integration into diverse task-specific codes.
- **Comprehensive Testing & Continuous Integration**: Achieves **99% code coverage** through rigorous testing with **pytest** and **Codecov**, coupled with continuous integration via **GitHub Actions**, ensuring software robustness and reliability.
- **Fast Prototyping & Algorithm Validation**: Streamlined support for defining, testing, and benchmarking new BLO algorithms.
- **Unified Computational Analysis**: Offers a comprehensive complexity analysis of gradient-based BLO techniques to guide users in selecting optimal configurations for efficiency and accuracy.
- **Detailed Documentation & Community Support**: Offers thorough documentation with practical examples and API references via **MkDocs**, ensuring accessibility and ease of use for both novice and advanced users.

##  🚀 **Why BOAT?**
Existing automatic differentiation (AD) tools primarily focus on specific and fundamental optimization strategies, such as explicit or implicit methods, and are often targeted at meta-learning or specific application scenarios, lacking support for algorithm customization. 

In contrast, **BOAT** expands the landscape of Bi-Level Optimization (BLO) applications by supporting a broader range of problem-adaptive operations. It bridges the gap between theoretical research and practical deployment, offering unparalleled flexibility to design, customize, and accelerate BLO techniques.


##  🏭 **Applications**
BOAT enables efficient implementation and adaptation of advanced BLO techniques for key applications, including but not limited to:
- **Hyperparameter Optimization (HO)**
- **Neural Architecture Search (NAS)**
- **Adversarial Training (AT)**
- **Few-Shot Learning (FSL)**
- **Medical Image Analysis (MIA)**
- **Generative Adversarial Learning**
- **Transfer Attack**
- ...

## 🔨 **Installation**

BOAT-jit is built on top of **Jittor**, so please **install Jittor first**, and then install **BOAT-jit**.

### 1. Install Jittor
You can follow the official installation guide:

👉 https://cg.cs.tsinghua.edu.cn/jittor/download/
#### Linux (Ubuntu / CentOS / Arch)

```bash
# 检查 python 版本大于等于 3.7
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
# 如果您电脑包含 Nvidia 显卡，检查 cuDNN 加速库
python3.7 -m jittor.test.test_cudnn_op
```
#### macOS

```bash
# 检查 python 版本大于等于 3.7
brew install libomp
python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
```
#### Windows

```bash
# 检查 python 版本大于等于 3.8
python --version
python -m pip install jittor
python -m jittor.test.test_core
python -m jittor.test.test_example
# 如果您电脑包含 Nvidia 显卡，检查 cuDNN 加速库
python -m jittor.test.test_cudnn_op
```

### 2. Install BOAT-jit
To install BOAT, use the following command:
```bash
pip install boat-jit 
or 
git clone -b boat_jit --single-branch https://github.com/callous-youth/BOAT.git
pip install -e .
```


##  ⚡ **How to Use BOAT**

### **1. Load Configuration Files**
BOAT relies on two key configuration files:
- `boat_config.json`: Specifies optimization strategies and dynamic/hyper-gradient operations.
- `loss_config.json`: Defines the loss functions for both levels of the BLO process.

```python
import os
import json
import boat_jit as boat

# Load configuration files
with open("path_to_configs/boat_config.json", "r") as f:
    boat_config = json.load(f)

with open("path_to_configs/loss_config.json", "r") as f:
    loss_config = json.load(f)
```

### **2. Define Models and Optimizers**
You need to specify both the upper-level and lower-level models along with their respective optimizers.

```python
import jittor as jit

# Define models
upper_model = UpperModel(*args, **kwargs)  # Replace with your upper-level model
lower_model = LowerModel(*args, **kwargs)  # Replace with your lower-level model

# Define optimizers
upper_opt = jit.nn.Adam(upper_model.parameters(), lr=0.01)
lower_opt = jit.nn.SGD(lower_model.parameters(), lr=0.01)
```

### **3. Customize BOAT Configuration**
Modify the boat_config to include your gradient mapping and numerical approximation operation, as well as model and variable details.

```python
# Example gradient mapping and numerical approximation operation Combination.
gm_op = ["NGD","DI", "GDA"]  # Gradient Mapping Operations (Demo Only)
na_op = ["RGT","RAD"]          # Numerical Approximation Operations (Demo Only)

# Add methods and model details to the configuration
boat_config["gm_op"] = gm_op
boat_config["na_op"] = na_op
boat_config["lower_level_model"] = lower_model
boat_config["upper_level_model"] = upper_model
boat_config["lower_level_var"] = lower_model.parameters()
boat_config["upper_level_var"] = upper_model.parameters()
```

### **4. Initialize the BOAT Problem**
Modify the boat_config to include your gradient mapping and numerical approximation operation, as well as model and variable details.

```python
# Initialize the problem
b_optimizer = boat.Problem(boat_config, loss_config)

# Build solvers for lower and upper levels
b_optimizer.build_ll_solver(lower_opt)  # Lower-level solver
b_optimizer.build_ul_solver(upper_opt)  # Upper-level solver
```

### **5. Define Data Feeds**
Prepare the data feeds for both levels of the BLO process, which was further fed into the the upper-level  and lower-level objective functions. 

```python
# Define data feeds (Demo Only)
ul_feed_dict = {"data": upper_level_data, "target": upper_level_target}
ll_feed_dict = {"data": lower_level_data, "target": lower_level_target}
```

### **6. Run the Optimization Loop**
Execute the optimization loop, optionally customizing the solver strategy for gradient mapping operations.

```python
# Set number of iterations
iterations = 1000

# Optimization loop (Demo Only)
for x_itr in range(iterations):
    # Run a single optimization iteration
    loss, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)

```



## Related Methods

- [Hyperparameter optimization with approximate gradient (CG)](https://arxiv.org/abs/1602.02355)
- [Optimizing millions of hyperparameters by implicit differentiation (NS)](http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (IAD)](https://arxiv.org/abs/1703.03400)
- [On First-Order Meta-Learning Algorithms (FOA)](https://arxiv.org/abs/1703.03400)
- [Bilevel Programming for Hyperparameter Optimization and Meta-Learning (RAD)](http://export.arxiv.org/pdf/1806.04910)
- [Truncated Back-propagation for Bilevel Optimization (RGT)](https://arxiv.org/pdf/1810.10667.pdf)
- [DARTS: Differentiable Architecture Search (FD)](https://arxiv.org/pdf/1806.09055.pdf)
- [A Generic First-Order Algorithmic Framework for Bi-Level Programming Beyond Lower-Level Singleton (GDA)](https://arxiv.org/pdf/2006.04045.pdf)
- [Towards gradient-based bilevel optimization with non-convex followers and beyond (PTT, DI)](https://proceedings.neurips.cc/paper_files/paper/2021/file/48bea99c85bcbaaba618ba10a6f69e44-Paper.pdf)
- [Averaged Method of Multipliers for Bi-Level Optimization without Lower-Level Strong Convexity(DM)](https://proceedings.mlr.press/v202/liu23y/liu23y.pdf)
- [Learning With Constraint Learning: New Perspective, Solution Strategy and Various Applications (IGA)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10430445)
- [BOME! Bilevel Optimization Made Easy: A Simple First-Order Approach (VFO)](https://proceedings.neurips.cc/paper_files/paper/2022/file/6dddcff5b115b40c998a08fbd1cea4d7-Paper-Conference.pdf)
- [A Value-Function-based Interior-point Method for Non-convex Bi-level Optimization (VSO)](http://proceedings.mlr.press/v139/liu21o/liu21o.pdf)
- [On Penalty-based Bilevel Gradient Descent Method (PGDO)](https://proceedings.mlr.press/v202/shen23c/shen23c.pdf)
- [Moreau Envelope for Nonconvex Bi-Level Optimization: A Single-loop and Hessian-free Solution Strategy (MESO)](https://arxiv.org/pdf/2405.09927)


## License

MIT License

Copyright (c) 2024 Yaohua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



