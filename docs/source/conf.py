# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys


from pathlib import Path


#<project_root>/boat_torch, <project_root>/docs/source/conf.py
CUR = Path(__file__).resolve()
PROJECT_ROOT = CUR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 将项目的根目录添加到 sys.path
sys.path.insert(0, os.path.abspath("../../"))

from unittest.mock import MagicMock
import sys

# mock mindspore
sys.modules["mindspore"] = MagicMock()

# mock boat_ms 和子模块
sys.modules["boat_ms"] = MagicMock()
sys.modules["boat_ms.utils"] = MagicMock()

# mock op_utils，并补上需要的函数
mock_op_utils = MagicMock()
mock_op_utils.copy_parameter_from_list = MagicMock()
mock_op_utils.require_model_grad = MagicMock()
mock_op_utils.l2_reg = MagicMock()
mock_op_utils.grad_unused_zero = MagicMock()

sys.modules["boat_ms.utils.op_utils"] = mock_op_utils




autodoc_typehints = "none"



html_logo = "_static/logo.jpg"

project = "BOAT-MS"
copyright = "2024, Yaohua Liu"
author = "Yaohua Liu"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx 配置
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # 支持 Google 和 NumPy 风格的 docstring
    "sphinx.ext.viewcode",  # 在文档中生成代码链接
    "myst_parser",  # 支持 Markdown (可选)
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",  # 引入自定义 CSS
]
# html_theme = 'alabaster'

html_context = {
    "extrahead": '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">',
}