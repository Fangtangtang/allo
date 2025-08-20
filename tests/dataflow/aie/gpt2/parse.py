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


SEQ_LEN = [ 256,512, 1024]
results = {}
for seq in SEQ_LEN:
    # total_time  = 0
    # for i in range(3):
        # with open(f"{seq}_{i}.txt", "r") as f:
        with open(f"../ffn_{seq}.txt", "r") as f:
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
        total_time  = 0
        for key, avg in averages.items():
            print(f"{key} average: {avg:.2f} us")
            total_time += avg
        results[seq] = total_time
        
    # results[seq] = total_time
print(results)
