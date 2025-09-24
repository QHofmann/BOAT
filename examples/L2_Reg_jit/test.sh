#!/bin/bash
SCRIPT="./l2_regularization_org.py"
COMMON_ARGS=""

# export jt_flags="use_cuda=0"
# export CUDA_VISIBLE_DEVICES=""   # 禁用 GPU
# export use_cuda=0                # 显式告诉 Jittor 不要用 CUDA
# export nvcc_path=""
# 通用参数
COMMON_ARGS=""

echo "================= 测试 Dynamic+Hyper 组合 ================="
dynamic_list=("NGD" "DI,NGD" "GDA,NGD" "GDA,NGD,DI" "DI,NGD,GDA")
hyper_list=("CG" "CG,PTT" "RAD" "RAD,PTT" "RAD,RGT" "PTT,RAD,RGT" "FD" "FD,PTT" "NS" "NS,PTT" "IGA" "IGA,PTT")

for dyn in "${dynamic_list[@]}"; do
  for hyp in "${hyper_list[@]}"; do
    echo "[RUN] dynamic=${dyn}, hyper=${hyp}"
    python $SCRIPT --dynamic_method $dyn --hyper_method $hyp $COMMON_ARGS || {
      echo "❌ Failed: dyn=$dyn hyp=$hyp"
      exit 1
    }
  done
done

echo "================= 测试 Dynamic+Hyper (DM) ================="
dynamic_dm_list=("DM,NGD" "DM,GDA,NGD")
hyper_dm_list=("RAD" "CG")

for dyn in "${dynamic_dm_list[@]}"; do
  for hyp in "${hyper_dm_list[@]}"; do
    echo "[RUN] dynamic=${dyn}, hyper=${hyp}"
    python $SCRIPT --dynamic_method $dyn --hyper_method $hyp $COMMON_ARGS || {
      echo "❌ Failed: dyn=$dyn hyp=$hyp"
      exit 1
    }
  done
done


echo "✅ 所有组合测试完成"