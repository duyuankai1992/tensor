import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置显示中文
matplotlib.rcParams['font.sans-serif'] = ['SemiHei'] #指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

N = 37
match_count = (392, 785, 1178, 1571, 1971, 2371, 2771, 3171, 3571, 3971, 4371, 4780, 5189, 5603, 6017, 6431, 6904, 7382, 7860, 8338, 8816, 9294, 9775, 10256, 10737, 11219, 11701, 12183, 12665, 13176, 13687, 14199, 14711, 15223, 15735, 16247, 16759)

match_count_delta = []
start = 0
for x in match_count:
    match_count_delta.append(x - start)
    start = x

x = np.arange(N)

plt.plot(x, match_count_delta, linewidth = 2)

# add some text for labels, title and axes ticks
plt.xlabel('Version')
plt.ylabel('Data Number')

#plt.legend()
plt.gca().yaxis.grid(True)

plt.show()