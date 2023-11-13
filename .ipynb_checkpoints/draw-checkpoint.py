import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#x方向
def draw_result(data, model, inner_radius_max):  
    # 训练与预测
    x = [float(i) for i in range(63-inner_radius_max, 63+inner_radius_max)]
    y = [58. for i in range(63-inner_radius_max, 63+inner_radius_max)]
    z = [29. for i in range(63-inner_radius_max, 63+inner_radius_max)]
    
    predict_values, ss = model.execute('points', x, y, z)
    plt.plot([i for i in range(63-inner_radius_max, 63+inner_radius_max)], predict_values ,color='b',linestyle="--",label="predict")
    
    plt.plot([i for i in range(63-inner_radius_max, 63+inner_radius_max)],[data[29][58][i] for i in range(63-inner_radius_max, 63+inner_radius_max)],color='r',linestyle="-",label='true')

    plt.xlabel("x")
    plt.ylabel("剂量")
    plt.title("x轴方向")
    fig = plt.gcf()
    fig.set_size_inches(15, 12)
    plt.legend(fontsize=12) 