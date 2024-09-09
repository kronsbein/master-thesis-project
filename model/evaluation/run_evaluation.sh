#!/bin/bash
# script to run evaluation of experiment results
# default values
data="c3o"
workloads=("grep" "kmeans" "pagerank" "sgd" "sort")
models=("TorchModel-f" "TorchModel-p" "TorchModel-s")
targets=("int" "ext")
experiments=("${data}-baseline" "${data}-ohe")
metric="sMAPE"

# parse command-line arguments
while getopts "d:w:m:t:e:x:" opt; do
    case $opt in
        d) data="$OPTARG" ;;
        w) IFS=',' read -r -a workloads <<< "$OPTARG" ;;
        m) IFS=',' read -r -a models <<< "$OPTARG" ;;
        t) IFS=',' read -r -a targets <<< "$OPTARG" ;;
        x) IFS=',' read -r -a experiments <<< "$OPTARG" ;;
        e) metric="$OPTARG" ;;
        *) echo "Usage: $0 [-d data] [-w workloads] [-m models] [-t targets] [-x experiments] [-e metric]" ;;
    esac
done

# loop through experiments
for experiment in "${experiments[@]}"; do
    echo "Start evaluation for: ${experiment}"
    echo "#################################"

    python runtime_prediction_analysis.py --data "${data}" \
    --directory ./${data}/experiment-"${experiment}" \
    --experiment "${experiment}" \
    --models "${models[@]}"
    
    for target in "${targets[@]}"; do
        python runtime_prediction_analysis.py --data "${data}" \
        --models "${models[@]}" \
        --workloads "${workloads[@]}" \
        --directory ./${data}/experiment-"${experiment}" \
        --experiment "${experiment}" \
        --target "${target}" \
        --metric "${metric}"
    done

    echo "#################################"
done