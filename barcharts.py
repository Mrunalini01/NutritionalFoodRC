import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import sys

def view(rlist):
    plt.rcdefaults()
    objects = ('BreakFast','Lunch','Dinner')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, rlist,color=['green','blue','brown'])
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.title('Analysis')
    plt.show()

