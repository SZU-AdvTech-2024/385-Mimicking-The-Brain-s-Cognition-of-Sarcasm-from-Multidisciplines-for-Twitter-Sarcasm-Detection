#!/bin/bash

# 定义参数数组
lr=0.0001
temps=(0.1 0.2 0.4 0.6 0.8)
#cl_weights=(0.1 0.2 0.4)
cl_weights=(0.6 0.8)

# GPU设备列表
gpu_devices=(0 1 2 3)

# 创建日志文件
log_file="screenlog2.txt"
echo "Screen Name and Script Parameters:" > ${log_file}

# 定义启动screen会话的函数
start_screen_session() {
    local screen_name=$1
    local temp=$2
    local cl_weight=$3
    local bce_weight=$4
    local gpu_device=$5
    local lr=$6

    # 将screen会话名称和参数写入日志文件
    echo "${screen_name}: CUDA_VISIBLE_DEVICES=${gpu_device} python train.py --lr ${lr} --temp ${temp} --cl-weight ${cl_weight} --bce-weight ${bce_weight}" >> ${log_file}

    # 创建并进入一个新的screen会话
    screen -S ${screen_name} -X quit
    screen -S ${screen_name} -dm bash -c "\
        conda activate llm; \
        cd /home/p/Documents/Codes/tweeter_sarcasm_detect/ablation/A2_cl; \
        CUDA_VISIBLE_DEVICES=${gpu_device} python train.py --lr ${lr} --temp ${temp} --cl-weight ${cl_weight} --bce-weight ${bce_weight}; exec bash"
}

# 初始化任务编号
task_num=1

# 循环参数数组，启动每个训练任务
for temp in "${temps[@]}"; do
    for cl_weight in "${cl_weights[@]}"; do
        # 计算bce-weight
        bce_weights=($(bc <<< "1-${cl_weight}") "1.0")
        for bce_weight in "${bce_weights[@]}"; do

            # 确定CUDA_VISIBLE_DEVICES的值
            gpu_device=${gpu_devices[$((task_num % 4))]}

            # 构造screen会话名称
            screen_name="t${task_num}"

            # 调用函数启动screen会话
            start_screen_session "${screen_name}" "${temp}" "${cl_weight}" "${bce_weight}" "${gpu_device}" "${lr}"

            # 递增任务编号
            ((task_num++))
        done
    done
done

echo "All training tasks started. Check ${log_file} for details."