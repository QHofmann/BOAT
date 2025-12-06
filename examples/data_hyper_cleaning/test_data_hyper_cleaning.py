import pytest
import subprocess
from unittest.mock import patch

gradient_mappinglist = (
    ["NGD"],
    ["DI", "NGD"],
    ["GDA", "NGD"],
    ["GDA", "NGD", "DI"],
    ["DI", "NGD", "GDA"],
)
numerical_approximationlist = (
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
gradient_mapping_dm = (["DM","NGD"], ["DM","GDA","NGD"])
numerical_approximation_dm = (["RAD"], ["CG"])
fo_ol_method = (["VSO"], ["VFO"], ["MESO"], ["PGDO"])


@pytest.mark.parametrize(
    "gradient_mapping, numerical_approximation",
    [
        (gradient_mapping, numerical_approximation)
        for gradient_mapping in gradient_mappinglist
        for numerical_approximation in numerical_approximationlist
    ],
)
def test_combination_dynamic_numerical_approximation(gradient_mapping, numerical_approximation):
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/data_hyper_cleaning/data_hyper_cleaning.py",
        "--gradient_mapping",
        ",".join(gradient_mapping),
        "--numerical_approximation",
        ",".join(numerical_approximation),
    ]
    print(
        f"Running test with gradient_mapping={gradient_mapping} and numerical_approximation={numerical_approximation}"
    )

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for gradient_mapping={gradient_mapping} and numerical_approximation={numerical_approximation}. Error: {result.stderr}"


@pytest.mark.parametrize(
    "gradient_mapping, numerical_approximation",
    [
        (gradient_mapping, numerical_approximation)
        for gradient_mapping in gradient_mapping_dm
        for numerical_approximation in numerical_approximation_dm
    ],
)
def test_combination_dynamic_numerical_approximation_dm(gradient_mapping, numerical_approximation):
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/data_hyper_cleaning/data_hyper_cleaning.py",
        "--gradient_mapping",
        ",".join(gradient_mapping),
        "--numerical_approximation",
        ",".join(numerical_approximation),
    ]
    print(
        f"Running test with gradient_mapping={gradient_mapping} and numerical_approximation={numerical_approximation}"
    )

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for gradient_mapping={gradient_mapping} and numerical_approximation={numerical_approximation}. Error: {result.stderr}"


@pytest.mark.parametrize("fo_ol_method", fo_ol_method)
def test_fo_ol_method(fo_ol_method):
    command = [
        "python",
        "/home/runner/work/BOAT/BOAT/examples/data_hyper_cleaning/data_hyper_cleaning.py",
        "--fo_op",
        fo_ol_method[0],
    ]
    print(f"Running test with fo_op={fo_ol_method}")

    result = subprocess.run(command, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"Test failed for fo_op={fo_ol_method}. Error: {result.stderr}"
