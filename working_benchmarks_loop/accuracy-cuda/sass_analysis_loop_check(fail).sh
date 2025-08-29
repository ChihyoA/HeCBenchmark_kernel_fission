cuobjdump --dump-sass main.o > kernel.sass

# 예: 특정 커널 이름이 _Z9myKernelv 라면 (mangled name, objdump는 mangled로 나옴)
kernel_name="_Z15accuracy_kerneliiiPKfPKiPi"

# Extract that kernel block into a temp file
awk "/Function : $kernel_name/,/^Function/" kernel.sass | head -n -1 > kernel_only.sass

sass_file="kernel_only.sass"

output_prefix="loop_body"  # e.g., loop_body_1.sass, loop_body_2.sass
loop_id=1

# 1. Find all backward branches → save (current_addr, target_addr) pairs
loop_ranges=$(awk '
  /^\/\*[0-9a-fA-F]+\*\// {
    match($0, /\/\*([0-9a-fA-F]+)\*\//, a);
    curr = strtonum("0x" a[1]);
    if ($0 ~ /BRA/ && match($0, /0x[0-9a-fA-F]+/, t)) {
      tgt = strtonum(t[0]);
      if (tgt < curr) {
        print "Loop detected from 0x" sprintf("%04x", tgt) " to 0x" sprintf("%04x", curr);
      }
    }
  }
' kernel_only.sass)


# 2. For each detected loop range, extract instructions and save
while IFS=',' read -r start_addr end_addr; do
  echo "Extracting loop from 0x$(printf '%04x' $start_addr) to 0x$(printf '%04x' $end_addr)..."

  awk -v start=$start_addr -v end=$end_addr '
    $1 ~ /^\/\*[0-9a-fA-F]+\*\// {
      split($1, a, "[/*]");
      addr = strtonum("0x" a[2]);
      if (addr >= start && addr <= end) print;
    }
  ' "$sass_file" > "${output_prefix}_${loop_id}.sass"

  echo "Saved to ${output_prefix}_${loop_id}.sass"
  loop_id=$((loop_id + 1))
done <<< "$loop_ranges"