import os
import time
import shutil

t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "hyper_cleaning\method_test"
gradient_mappinglist = (
    ["NGD"],
    ["DI", "NGD"],
    ["GDA", "NGD"],
    ["GDA", "NGD", "DI"],
    ["DI", "NGD", "GDA"],
)
gradient_mapping_dm = (["DM"], ["DM", "GDA"])
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
# m='Darts_W_RHG'
folder = "C:/Users/ASUS/Documents/GitHub/BOAT/examples/data_hyper_cleaning"
folder = os.path.join(folder, args, t0)

print(folder)
if not os.path.exists(folder):
    os.makedirs(folder)
batfolder = os.path.join(folder, "set.bat")
ganfolder = os.path.join(folder, "data_hyper_cleaning.py")
shutil.copyfile("data_hyper_cleaning.py", ganfolder)
utilfolder = os.path.join(folder, "util_file.py")
shutil.copyfile("util_file.py", utilfolder)
with open(batfolder, "w") as f:
    k = 0
    for gradient_mapping in gradient_mappinglist:
        for numerical_approximation in numerical_approximationlist:
            k += 1
            print("Comb.{}:".format(k))
            print("gradient_mapping:", gradient_mapping, " numerical_approximation:", numerical_approximation)
            f.write(
                "python data_hyper_cleaning.py --gradient_mapping {} --numerical_approximation {} \n".format(
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
                "python data_hyper_cleaning.py --gradient_mapping {} --numerical_approximation {} \n".format(
                    ",".join([dynamic for dynamic in gradient_mapping]),
                    ",".join([hyper for hyper in numerical_approximation]),
                )
            )
    for numerical_approximation in fo_ol_method:
        k += 1
        print("Comb.{}:".format(k))
        print("numerical_approximation:", numerical_approximation)
        f.write("python data_hyper_cleaning.py --fo_op {} \n".format(numerical_approximation[0]))
# os.chdir(folder)
os.system(batfolder)

# import os
# import time
# import shutil
# import platform
#
# # 获取当前时间
# t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
# args = 'hyper_cleaning/method_test'  # 使用相对路径
# gradient_mappinglist = (["NGD"], ["DI", "NGD"], ["GDA", "NGD", "DI"], ["DI", "NGD", "GDA"])
# gradient_mapping_dm = (["NGD", "DM"], ["NGD", "DM", "GDA"])
# numerical_approximationlist = (
#     ["CG"], ["CG", "PTT"], ["RAD"], ["RAD", "PTT"], ["RAD", "RGT"], ["PTT", "RAD", "RGT"], ["FD"], ["FD", "PTT"], ["NS"],
#     ["NS", "PTT"], ["IGA"]
# )
# numerical_approximation_dm = (["RAD"], ["CG"])
# fo_ol_method = (["VSO"], ["VFO"], ["MESO"], ["PGDO"])
#
# # 获取当前脚本所在的目录（相对路径）
# base_folder = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
# folder = os.path.join(base_folder, args, t0)  # 构建相对路径
#
# print(folder)
# if not os.path.exists(folder):
#     os.makedirs(folder)
#
# # 修改 bat 文件为 sh 文件
# script_extension = '.bat' if platform.system() == "Windows" else '.sh'
# script_file = os.path.join(folder, 'set' + script_extension)
#
# # 将 Python 文件复制到目标文件夹
# ganfolder = os.path.join(folder, 'data_hyper_cleaning.py')
# shutil.copyfile(os.path.join(base_folder,'data_hyper_cleaning.py'), ganfolder)
# utilfolder = os.path.join(folder, 'util_file.py')
# shutil.copyfile(os.path.join(base_folder,'util_file.py'), utilfolder)
#
# # 创建批处理或 shell 脚本
# with open(script_file, 'w') as f:
#     k = 0
#     for gradient_mapping in gradient_mappinglist:
#         for numerical_approximation in numerical_approximationlist:
#             k += 1
#             print("Comb.{}:".format(k))
#             print('gradient_mapping:', gradient_mapping, ' numerical_approximation:', numerical_approximation)
#             f.write('python data_hyper_cleaning.py --gradient_mapping {} --numerical_approximation {} \n'.format(
#                 ','.join([dynamic for dynamic in gradient_mapping]), ','.join([hyper for hyper in numerical_approximation])))
#
#     for gradient_mapping in gradient_mapping_dm:
#         for numerical_approximation in numerical_approximation_dm:
#             k += 1
#             print("Comb.{}:".format(k))
#             print('gradient_mapping:', gradient_mapping, ' numerical_approximation:', numerical_approximation)
#             f.write('python data_hyper_cleaning.py --gradient_mapping {} --numerical_approximation {} \n'.format(','.join(
#                 [dynamic for dynamic in gradient_mapping]), ','.join([hyper for hyper in numerical_approximation])))
#
#     for numerical_approximation in fo_ol_method:
#         k += 1
#         print("Comb.{}:".format(k))
#         print('numerical_approximation:', numerical_approximation)
#         f.write('python data_hyper_cleaning.py --fo_op {} \n'.format(numerical_approximation[0]))
#
# # 如果是 Ubuntu 系统, 使得脚本具有执行权限
# if platform.system() != "Windows":
#     os.chmod(script_file, 0o775)  # 给 sh 文件执行权限
#
# print('right!')
#
# # 切换到指定文件夹并运行脚本
# os.chdir(folder)
# if platform.system() == "Windows":
#     os.system(script_file)  # Windows 下运行 .bat 文件
# else:
#     os.system(f"bash {script_file}")  # Ubuntu 下运行 .sh 文件
