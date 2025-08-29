#!/bin/bash

# 현재 폴더 내 모든 하위 디렉토리를 탐색
find . -mindepth 1 -type d | while read -r dir; do
    # Makefile이 존재하는지 확인
    if [[ -f "$dir/Makefile" ]]; then
        echo "Modifying Makefile in: $dir"
        
        # Makefile의 마지막 이전 줄 가져오기 (A)
        last_before_line=$(tail -n 1 "$dir/Makefile")

        # 새로운 run_nsys 타겟 추가
        echo -e "\nrun_nsys:\n\tnsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node $last_before_line" >> "$dir/Makefile"
        
        echo "Updated $dir/Makefile"
    fi
done