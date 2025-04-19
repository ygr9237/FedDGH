#!/bin/bash

if [[ ! "$0" =~ /tmp/test_[0-9]{8}_[0-9]{6}\.sh ]]; then
    BACKUP_SCRIPT="/tmp/test_$(date +%Y%m%d_%H%M%S).sh"
    cp "$0" "$BACKUP_SCRIPT"
    chmod +x "$BACKUP_SCRIPT"
    echo "created: $BACKUP_SCRIPT"

    echo "$BACKUP_SCRIPT" | at now + 1 minutes
    exit 0
fi

function start_fed_task() {
    local method=$1
    local method_dir=$2
    local device=$3
    local port=$4
    local num_rounds=$5
    local address=$6
    local dataset=$7
    local num_classes=$8
    local target=$9
    local log_prefix=${10}
    shift 10
    local subdatasets=("$@")

    local server="${method}_server.py"
    local client="${method}_client.py"

    source /home/yiyue/anaconda3/bin/activate yiyue_envs   # replace graph and your envs_name
    cd /workspace/yiyue/FedDGH/source/main || exit 1  # replace your file_name

    export CUDA_VISIBLE_DEVICES=$device

    mkdir -p "../ckps/${method_dir}/${dataset}"

    nohup python $server \
        --server_port=$port \
        --dataset="$dataset" \
        --server_address="$address" \
        --target="$target" \
        --num_rounds=$num_rounds \
        --num_classes=$num_classes \
        --eta=0.7 \
        --alfa=0.01 \
        --beta=0.5 \
        --restore=0 \
        --code_length=64 2>&1 | tee -a "../ckps/${method_dir}/${dataset}/${log_prefix}.log"&

    sleep 10

    for idx in "${!subdatasets[@]}"; do
        client_num=$((idx + 1))
        nohup python $client \
            --server_port=$port \
            --dataset="$dataset" \
            --server_address="$address" \
            --subdataset="${subdatasets[idx]}" \
            --client="$client_num" \
            --num_classes=$num_classes \
            --eta=0.7 \
            --alfa=0.01 \
            --code_length=64 2>&1 | tee -a "../ckps/${method_dir}/${dataset}/${log_prefix}_nohup.log"&
        sleep 3
    done
}

start_fed_task "hx" "FedDGH" 0 9934 50 "127.0.0.1"  "PACS" 7 "art_painting" "A" "cartoon" "photo" "sketch"
start_fed_task "hx" "FedDGH" 1 9935 50 "127.0.0.1" "PACS" 7 "cartoon"      "C" "art_painting" "photo" "sketch"
#start_fed_task "hx" "FedDGH" 0 9930 50 "127.0.0.1"  "PACS" 7 "photo"       "P" "art_painting" "cartoon" "sketch"
#start_fed_task "hx" "FedDGH" 1 9942 50 "127.0.0.1" "PACS" 7 "sketch"      "S" "art_painting" "cartoon" "photo"

#start_fed_task "hx" "FedDGH" 0 9910 50 "127.0.0.1"  "OfficeHome" 65 "Art"        "A" "Clipart" "Product" "Real_World"
#start_fed_task "hx" "FedDGH" 1 9911 50 "127.0.0.1" "OfficeHome" 65 "Clipart"    "C" "Art" "Product" "Real_World"
#start_fed_task "hx" "FedDGH" 0 9912 50 "127.0.0.1"  "OfficeHome" 65 "Product"    "P" "Art" "Clipart" "Real_World"
#start_fed_task "hx" "FedDGH" 1 9913 50 "127.0.0.1" "OfficeHome" 65 "Real_World" "R" "Art" "Clipart" "Product"
#
#start_fed_task "hx" "FedDGH" 0 9920 50 "127.0.0.1"  "VLCS" 5 "Caltech101" "C" "LabelMe" "SUN09" "VOC2007"
#start_fed_task "hx" "FedDGH" 1 9921 50 "127.0.0.1" "VLCS" 5 "LabelMe"    "L" "Caltech101" "SUN09" "VOC2007"
#start_fed_task "hx" "FedDGH" 0 9922 50 "127.0.0.1"  "VLCS" 5 "SUN09"      "S" "Caltech101" "LabelMe" "VOC2007"
#start_fed_task "hx" "FedDGH" 1 9923 50 "127.0.0.1" "VLCS" 5 "VOC2007"    "V" "Caltech101" "LabelMe" "SUN09"

#echo "/workspace/yiyue/FedDGH/source/main/test.sh" | at now + 1 minutes
#chmod +x /workspace/yiyue/FedDGH/source/main/test.sh
#127.0.0.1

