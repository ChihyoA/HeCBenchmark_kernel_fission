#!/bin/bash

# 현재 폴더 내 모든 하위 디렉토리를 탐색
find . -mindepth 1 -type d | while read -r dir; do
    # Makefile이 존재하는지 확인
    if [[ -f "$dir/Makefile" ]]; then
        echo "Entering directory: $dir"
        cd "$dir" || continue

        # make 실행
        echo "Running make in $dir..."
        make &> make_result.txt
        make_status=$?  # make 실행 결과 저장

        # make 성공 여부 확인
        if [[ $make_status -eq 0 ]]; then
            echo "Make completed successfully."

            # Makefile의 마지막 5줄을 확인하여 "run_nsys:"이 포함되어 있는지 검사
            last_lines=$(tail -n 5 Makefile)
            if [[ "$last_lines" == *"run_nsys:"* ]]; then
                echo "Found run_nsys: in Makefile. Running make run_nsys..."
                make run_nsys &> result.txt
            else
                echo "run_nsys target not found in Makefile. Skipping make run_nsys."
            fi
        else
            echo "Make failed in $dir. Skipping make run_nsys."
        fi

        echo "Completed $dir"
        cd - > /dev/null  # 원래 위치로 돌아가기
    fi
done