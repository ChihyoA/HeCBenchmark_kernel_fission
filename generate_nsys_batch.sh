#!/bin/bash

# 디렉토리 리스트
directories=("cuda_m" "cuda_nop" "cuda_qr" "cuda_s" "cuda_tz")

# 각 디렉토리에 들어가서 ./generate_nsys_report.sh 실행
for dir in "${directories[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "Entering directory: $dir"
        cd "$dir" || continue

        # 실행 가능한 generate_nsys_report.sh가 있는지 확인
        if [[ -x "./generate_nsys_report.sh" ]]; then
            echo "Running ./generate_nsys_report.sh in $dir..."
            ./generate_nsys_report.sh
        else
            echo "Error: ./generate_nsys_report.sh not found or not executable in $dir."
        fi

        cd - > /dev/null  # 원래 위치로 돌아가기
    else
        echo "Error: Directory $dir does not exist."
    fi
done

echo "All directories processed."