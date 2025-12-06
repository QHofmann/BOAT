import os
import time
import shutil
import platform

t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "l2_regularization/method_test"
gradient_mappinglist = (
    ["NGD"],
    ["DI", "NGD"],
    ["GDA", "NGD"],
    ["GDA", "NGD", "DI"],
    ["DI", "NGD", "GDA"],
)
gradient_mapping_dm = (["NGD", "DM"], ["NGD", "DM", "GDA"])
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
numerical_approximation_dm = (["RAD"], ["CG"])
fo_ol_method = (["VSO"], ["VFO"], ["MESO"], ["PGDO"])

base_folder = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base_folder, args, t0)

print(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

script_extension = ".bat" if platform.system() == "Windows" else ".sh"
script_file = os.path.join(folder, "set" + script_extension)


ganfolder = os.path.join(folder, "l2_regularization.py")
shutil.copyfile(os.path.join(base_folder, "l2_regularization.py"), ganfolder)
with open(script_file, "w") as f:
    k = 0
    for gradient_mapping in gradient_mappinglist:
        for numerical_approximation in numerical_approximationlist:
            k += 1
            print("Comb.{}:".format(k))
            print("gradient_mapping:", gradient_mapping, " numerical_approximation:", numerical_approximation)
            f.write(
                "python l2_regularization.py --gradient_mapping {} --numerical_approximation {} \n".format(
                    ",".join([dynamic for dynamic in gradient_mapping]),
                    ",".join([hyper for hyper in numerical_approximation]),
                )
            )

    for gradient_mapping in gradient_mapping_dm:
        for numerical_approximation in numerical_approximation_dm:
            k += 1
            print("Comb.{}:".format(k))
            print("gradient_mapping:", gradient_mapping, " numerical_approximation:", numerical_approximation)
            f.write(
                "python l2_regularization.py --gradient_mapping {} --numerical_approximation {} \n".format(
                    ",".join([dynamic for dynamic in gradient_mapping]),
                    ",".join([hyper for hyper in numerical_approximation]),
                )
            )

    for numerical_approximation in fo_ol_method:
        k += 1
        print("Comb.{}:".format(k))
        print("numerical_approximation:", numerical_approximation)
        f.write("python l2_regularization.py --fo_op {} \n".format(numerical_approximation[0]))

if platform.system() != "Windows":
    os.chmod(script_file, 0o775)

print("right!")

# os.chdir(folder)
if platform.system() == "Windows":
    os.system(script_file)
else:
    os.system(f"bash {script_file}")
