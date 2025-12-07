import os
import time
import shutil
import platform

t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "meta_learning\method_test"
# gm_oplist=(["NGD"],["DI","NGD"],["GDA","NGD","DI"],["DI","NGD","GDA"])
# gm_op_dm = (["NGD","DM"],["NGD","DM","GDA"])
# na_oplist = (["CG"],["CG","PTT"],["RAD"],["RAD","PTT"],["RAD","RGT"],["PTT","RAD","RGT"],["FD"],["FD","PTT"],["NS"],["NS","PTT"],["IGA"])
# na_op_dm = (["RAD"],["CG"])
gm_oplist = (["NGD"], ["NGD", "GDA"])
na_oplist = (
    ["IAD"],
    ["IAD", "PTT"],
    ["CG", "IAD"],
    ["CG", "IAD", "PTT"],
    ["NS", "IAD"],
    ["NS", "IAD", "PTT"],
    ["FOA", "IAD"],
    ["FOA", "IAD", "PTT"],
)
# na_op_dm = (["RAD"],["CG"])
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
    for gm_op in gm_oplist:
        for na_op in na_oplist:
            k += 1
            print("Comb.{}:".format(k))
            print("gm_op:", gm_op, " na_op:", na_op)
            f.write(
                "python meta_learning.py --gm_op {} --na_op {} \n".format(
                    ",".join([dynamic for dynamic in gm_op]),
                    ",".join([hyper for hyper in na_op]),
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
