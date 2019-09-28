from _4_2 import getBestValidationScore
import numpy as np
import time


# result2l = [getBestValidationScore((2,), 0.05, noise_sigma=0.09) for _ in range(20)]
# result3l = [
#     getBestValidationScore((2, 3), 0, noise_sigma=0.09, learn_rate=0.1)
#     for _ in range(20)
# ]

# print(
#     f"""
# 2 Layer Results:
# Mean: {np.mean(result2l)}
# Std: {np.std(result2l)}
# 3 Layer Results:
# Mean: {np.mean(result3l)}
# Std: {np.std(result3l)}
# """
# )

time2l = []
time3l = []
vscore = []
for i in range(10):
    start = time.time()
    vscore.append(getBestValidationScore((2), 0.05))
    time2l.append(time.time()-start)
    
    start = time.time()
    # getBestValidationScore((2, 3), 0, noise_sigma=0.09, learn_rate=0.1)
    time3l.append(time.time()-start)

print (np.mean(vscore), np.std(vscore))

print(
    f"""
2 Layer Results:
Mean: {np.mean(time2l)}
Std: {np.std(time2l)}
3 Layer Results:
Mean: {np.mean(time3l)}
Std: {np.std(time3l)}
"""
)