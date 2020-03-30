import time

def benchmark(func):
    counter = 100
    def inner(*args, **kwargs):
        durations = []
        for i in range(counter):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            durations.append((end - start) * 1000)
        durations.pop(0)
        ave = sum(durations) / counter
        min_value = min(durations)
        max_value = max(durations)
        print(f'average:{round(ave, 4)}[ms] min:{round(min_value, 4)}, max:{round(max_value, 4)}')
        return func(*args, **kwargs)
    return inner
