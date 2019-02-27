import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.savitzky_golay import savitzky_golay

unrate = pd.read_csv('/home/tony/Downloads/run_.-tag-cost_validation.csv')
unrate2 = pd.read_csv('/home/tony/Downloads/run_.-tag-cost_training.csv')

first_1k = unrate[0:1000]
newX = np.asarray(first_1k['Step'])
newY = np.asarray(first_1k['Value'])

first_1k_train = unrate2[0:1000]
newY_train = np.asarray(first_1k_train['Value'])
newY_train = savitzky_golay(newY_train, window_size=41, order=3)

plotX = []
plotY = []
currentSum = 0
currentNum = 0
currentStep = 3250
for x in range(len(newX)):
    print(x)
    if newX[x] == currentStep:
        currentSum += newY[x]
        currentNum += 1
    else:
        print(currentStep)
        plotX.append(currentStep)
        currentStep = newX[x]
        plotY.append(currentSum / currentNum)
        currentSum = newY[x]
        currentNum = 1


newY = savitzky_golay(newY, window_size=41, order=3)

plt.plot(plotX, plotY, label="Validation Acc", marker='+')
plt.plot(first_1k_train['Step'], newY_train, color="red", label="Training Acc")

num = 0
for a, b in zip(plotX, plotY):
    num += 1
    if num == len(plotX):
        plt.annotate(format(b, '.4f'), xy=(a, b), xytext=(-30, -30), textcoords='offset points',
                     fontsize=15, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

num = 0
for a, b in zip(first_1k_train['Step'], newY_train):
    num += 1
    if num == len(newY_train):
        plt.annotate(format(b, '.4f'), xy=(a, b), xytext=(-30, -38), textcoords='offset points',
                     fontsize=15, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.legend()

plt.xticks(rotation=30)
plt.xlabel('Global Steps')
plt.ylabel('Accuracy')
plt.title('GAN / Training and Validation Accuracy')
# plt.show()
plt.savefig("Acc-GAN.jpg", bbox_inches='tight')
