cuobjdump --dump-sass kernel.o > kernel.sass

# 예: 특정 커널 이름이 _Z9myKernelv 라면 (mangled name, objdump는 mangled로 나옴)
kernel_name="_Z7kernel1P12ComplexFloatS0_S0_ff"

# Extract that kernel block into a temp file
awk "/Function : $kernel_name/,/^Function/" kernel.sass | head -n -1 > kernel_only.sass

sass_file="kernel_only.sass"

# Match any line that looks like a SASS instruction (e.g., starts with /*xxxx*/ and has an opcode)
total_instr=$(grep -E '^[[:space:]]*/\*[0-9a-f]+\*/[[:space:]]+[A-Z]' "$sass_file" | wc -l)

# Memory instructions: load/store global/local/shared
#mem_instr=$(grep -E '\b(LDG|STG|REDG|LDS|STS|LDL|STL|RED|GLOBAL|SHARED)\b' "$sass_file" | wc -l)
mem_instr=$(grep -E '\b(LDG|STG|REDG|GLOBAL)\b' "$sass_file" | wc -l)

# Compute instructions: arithmetic/logical/int/float/math ops (you can refine as needed)
#compute_instr=$(grep -E '\b(IMAD|IADD|FADD|FMUL|FFMA|VIADD|ISETP|FSETP|HFMA|MOV|SHF|LEA|MMA|UFLO|VOTE|PLOP3)\b' "$sass_file" | wc -l)
#compute_instr=$(grep -E '\b(FFMA|FMUL|IMAD|MMA|LEA|SHF|HFMA|FSETP|ISETP)\b' "$sass_file" | wc -l)
compute_instr=$(grep -E '\b(FFMA|MMA|HFMA|IMAD|LEA)\b' "$sass_file" | wc -l)

# Atomic / Reduction-like instr
atomic_instr=$(grep -E '\b(REDG|REDUX|ATOM|ATOMIC|RED|STRONG)\b' "$sass_file" | wc -l)

# Output percentages
mem_ratio=$(awk -v m="$mem_instr" -v t="$total_instr" 'BEGIN { printf "%.2f", (t > 0 ? m / t * 100 : 0) }')
compute_ratio=$(awk -v c="$compute_instr" -v t="$total_instr" 'BEGIN { printf "%.2f", (t > 0 ? c / t * 100 : 0) }')
atomic_ratio=$(awk -v a="$atomic_instr" -v t="$total_instr" 'BEGIN { printf "%.2f", (t > 0 ? a / t * 100 : 0) }')

echo "Total instructions     : $total_instr"
echo "Memory instructions    : $mem_instr ($mem_ratio%)"
echo "Compute instructions   : $compute_instr ($compute_ratio%)"
echo "Atomic instructions   : $atomic_instr ($atomic_ratio%)"