import re
import statistics

def process_data(data, k=1):
    # avg, std
    mu = statistics.mean(data)
    sigma = statistics.pstdev(data)
    # range
    lower = mu - k * sigma
    upper = mu + k * sigma
    return [x for x in data if lower <= x <= upper]

with open("gemm_opt.txt", "r") as f:
    text = f.read()

patterns = {
    "Context creation time": r"Context creation time:\s*(\d+)us",
    "Inst time": r"Inst time:\s*(\d+)us",
    "Copy time": r"Copy time:\s*(\d+)us",
    "NPU execution time": r"NPU execution time:\s*(\d+)us",
}

averages = {}
for key, pattern in patterns.items():
    matches = re.findall(pattern, text)
    values = list(map(int, matches))
    if values:
        values = process_data(values)
        averages[key] = sum(values) / len(values)

for key, avg in averages.items():
    print(f"{key} average: {avg:.2f} us")

"""
gemm_wo_vm.txt
Context creation time average: 14544.30 us
Inst time average: 129.84 us
Copy time average: 88.56 us
NPU execution time average: 3413.71 us
gemm_w_vm.txt
Context creation time average: 14976.80 us
Inst time average: 148.45 us
Copy time average: 143.92 us
NPU execution time average: 6419.11 us
gemm_vec.txt
Context creation time average: 15096.53 us
Inst time average: 126.68 us
Copy time average: 142.59 us
NPU execution time average: 3133.56 us
gemm_opt.txt
Context creation time average: 13998.96 us
Inst time average: 138.06 us
Copy time average: 153.18 us
NPU execution time average: 2825.92 us
"""