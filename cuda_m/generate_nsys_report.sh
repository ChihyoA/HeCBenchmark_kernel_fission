#!/bin/bash

# CSV 파일 초기화 (헤더 추가)
csv_file="results.csv"
echo "Directory Name,Make Success,run_nsys Found,run_nsys Success" > "$csv_file"

# 현재 폴더 내 모든 하위 디렉토리를 탐색
find . -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    # Makefile이 존재하는지 확인
    if [[ -f "$dir/Makefile" ]]; then
        echo "Entering directory: $dir"
        cd "$dir" || continue

        # make 실행
        echo "Running make in $dir..."
        make
        make_status=$?  # make 실행 결과 저장
        make_result="FAIL"

        if [[ $make_status -eq 0 ]]; then
            make_result="SUCCESS"
        fi

        # Makefile의 마지막 5줄을 확인하여 "run_nsys:"이 포함되어 있는지 검사
        last_lines=$(tail -n 5 Makefile)
        run_nsys_found="NO"
        run_nsys_result="N/A"

        if [[ "$last_lines" == *"run_nsys:"* ]]; then
            run_nsys_found="YES"

            if [[ $make_status -eq 0 ]]; then
                echo "Found run_nsys: in Makefile. Running make run_nsys..."
                make run_nsys &> result.txt
                run_nsys_status=$?

                if [[ $run_nsys_status -eq 0 ]]; then
                    run_nsys_result="SUCCESS"
                else
                    run_nsys_result="FAIL"
                fi
            fi
        fi

        # 결과를 CSV 파일에 추가
        echo "\"$dir\",$make_result,$run_nsys_found,$run_nsys_result" >> "../$csv_file"

        echo "Completed $dir"
        cd - > /dev/null  # 원래 위치로 돌아가기
    fi
done

echo "All iterations completed. Results saved in $csv_file"