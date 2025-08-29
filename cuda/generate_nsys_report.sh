#!/bin/bash

# 현재 폴더 내 모든 하위 디렉토리를 탐색
find . -mindepth 1 -type d | while read -r dir; do
    # Makefile이 존재하는지 확인
    if [[ -f "$dir/Makefile" ]]; then
        echo "Entering directory: $dir"
        cd "$dir" || continue

        # make 실행
        echo "Running make in $dir..."
        make

        # Makefile의 마지막 줄을 확인하여 "$(program)"이 포함되어 있는지 검사
        last_line=$(tail -n 1 Makefile)
        if [[ "$last_line" == *"\$(program)"* ]]; then
            echo "Found \$(program) in Makefile. Running make run_nsys..."
            make run_nsys &> result.txt
        else
            echo "\$(program) not found in Makefile. Skipping make run_nsys."
        fi

        echo "Completed $dir"
        cd - > /dev/null  # 원래 위치로 돌아가기
    fi
done