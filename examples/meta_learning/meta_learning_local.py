import os
import time
import shutil
import platform

t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "meta_learning\method_test"
# gradient_mappinglist=(["NGD"],["DI","NGD"],["GDA","NGD","DI"],["DI","NGD","GDA"])
# gradient_mapping_dm = (["NGD","DM"],["NGD","DM","GDA"])
# numerical_approximationlist = (["CG"],["CG","PTT"],["RAD"],["RAD","PTT"],["RAD","RGT"],["PTT","RAD","RGT"],["FD"],["FD","PTT"],["NS"],["NS","PTT"],["IGA"])
# numerical_approximation_dm = (["RAD"],["CG"])
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
# numerical_approximation_dm = (["RAD"],["CG"])
# m='Darts_W_RHG'
base_folder = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base_folder, args, t0)

print(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

script_extension = ".bat" if platform.system() == "Windows" else ".sh"
script_file = os.path.join(folder, "set" + script_extension)
if not os.path.exists(folder):
    os.makedirs(folder)

script_extension = ".bat" if platform.system() == "Windows" else ".sh"
script_file = os.path.join(folder, "set" + script_extension)

ganfolder = os.path.join(folder, "meta_learning.py")
shutil.copyfile(os.path.join(base_folder, "meta_learning.py"), ganfolder)
with open(script_file, "w") as f:
    k = 0
    for gradient_mapping in gradient_mappinglist:
        for numerical_approximation in numerical_approximationlist:
            k += 1
            print("Comb.{}:".format(k))
            print("gradient_mapping:", gradient_mapping, " numerical_approximation:", numerical_approximation)
            f.write(
                "python meta_learning.py --gradient_mapping {} --numerical_approximation {} \n".format(
                    ",".join([dynamic for dynamic in gradient_mapping]),
                    ",".join([hyper for hyper in numerical_approximation]),
                )
            )


if platform.system() != "Windows":
    os.chmod(script_file, 0o775)

print("right!")

# os.chdir(folder)
if platform.system() == "Windows":
    os.system(script_file)
else:
    os.system(f"bash {script_file}")
