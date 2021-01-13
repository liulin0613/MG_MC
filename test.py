import re

from matplotlib import pyplot as plt

f = open("./result/yelp4/l2.out")
line = f.readline()
rmse = []
mae = []
while line:
    if line.startswith("<Test>"):
        value = re.findall(r'\d+\.\d+', line)
        if float(value[0]) < 1:
            rmse.append(float(value[0]))
            mae.append(float(value[1]))
    line = f.readline()

f.close()

alix = range(1, len(rmse) + 1)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # 做镜像处理
p1 = ax1.plot(alix, rmse, 'g-')
p2 = ax2.plot(alix, mae, 'r--')
ax1.axhline(0.3806, c="g", ls="--", lw=0.5)
ax2.axhline(0.1029, c="r", ls="--", lw=0.5)
ax1.set_xlabel('epoch')
ax1.set_ylabel('RMSE', color='g')
ax2.set_ylabel('MAE', color='r')
ax1.legend(["RMSE"], loc='upper center')
plt.legend(["MAE"])

plt.show()
