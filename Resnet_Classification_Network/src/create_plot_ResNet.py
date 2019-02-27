import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.savitzky_golay import savitzky_golay

unrate = pd.read_csv('/home/tony/Downloads/run_.-tag-cost_training.csv')
unrate2 = pd.read_csv('/home/tony/Downloads/run_.-tag-cost_validation.csv')

first_1k = unrate[0:1000]
first_1k_val = unrate2[0:1000]
newY = np.asarray(first_1k['Value'])
newY_val = np.asarray(first_1k_val['Value'])
newY = savitzky_golay(newY, window_size=41, order=3)
newY_val = savitzky_golay(newY_val, window_size=41, order=3)

plt.plot(first_1k['Step'], newY, color="red", label="Training Loss")
plt.plot(first_1k_val['Step'], newY_val, label="Validation Loss", marker='+')
plt.legend()

plt.xticks(rotation=30)
plt.xlabel('Global Steps')
plt.ylabel('Loss (MSE)')
plt.title('CNNs and Inception-RestNet-v2 / Loss')
# plt.show()
plt.savefig("Loss-CNNs-Inception-ResNet-v2.jpg")
