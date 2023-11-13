import pandas as pd
import numpy as np
import random

# 读取每个表格的数据并存储在字典中
def get_data(data_sheet_name):
    data = {}
    for z in range(72):
        sheet_name = f"avg_1_{z+1}"
        df = pd.read_excel(data_sheet_name, sheet_name=sheet_name, header=None, usecols=("B:EG"), names=list(range(136)))
        data[z] = df
    return data


def training_sampling(data, center_x, center_y, center_z, inner_radius_list, inner_radius_max):
    # 计算立方体的八个顶点坐标
    vertices = [item for sublist in
        [[(center_x + inner_radius, center_y + inner_radius, center_z + inner_radius),
        (center_x + inner_radius, center_y + inner_radius, center_z - inner_radius),
        (center_x + inner_radius, center_y - inner_radius, center_z + inner_radius),
        (center_x + inner_radius, center_y - inner_radius, center_z - inner_radius),
        (center_x - inner_radius, center_y + inner_radius, center_z + inner_radius),
        (center_x - inner_radius, center_y + inner_radius, center_z - inner_radius),
        (center_x - inner_radius, center_y - inner_radius, center_z + inner_radius),
        (center_x - inner_radius, center_y - inner_radius, center_z - inner_radius)]
        for inner_radius in inner_radius_list] for item in sublist
    ]
    
    # 取样八个顶点的数据
    sampled_data = []
    for vertex in vertices:
        x, y, z = vertex
        sampled_value = data[z][y][x]
        sampled_data.append((x, y, z, sampled_value))
    # 
    step_sizes = [2,3]  # 步长列表

    ## 根据给定的步长和点的数量提取数据点
    for step in step_sizes:
        x_range = inner_radius_max // step
        y_range = inner_radius_max // step
        z_range = inner_radius_max // step

        for x in range(-x_range + 1, x_range + 1):
            for y in range(-y_range + 1, y_range + 1):
                for z in range(-z_range + 1, z_range + 1):
                    x_coord = center_x+x * step
                    y_coord = center_y+y * step
                    z_coord = center_z+z * step

                    value = data[z_coord][y_coord][x_coord]
                    sampled_data.append((x_coord,y_coord,z_coord,value))

    
    # 添加源点（弃用/no）
    sampled_data.append((63,58,29,data[29][58][63]))
    
    # 改格式为dataFrame
    sampled_df = pd.DataFrame(sampled_data, columns=['x', 'y', 'z', 'target'])
    print("Training Sampled DataFrame:")
    return sampled_df

def testing_sampling(data, center_x, center_y, center_z, inner_radius, test_nums):
    vertices = [(random.randint(center_x-inner_radius, center_x+inner_radius), random.randint(center_y-inner_radius, center_y+inner_radius), random.randint(center_z-inner_radius, center_z+inner_radius)) for i in range(test_nums)]
    
    sampled_data = []
    for vertex in vertices:
        x, y, z = vertex
        sampled_value = data[z][y][x]
        sampled_data.append((x, y, z, sampled_value))
        
    sampled_df = pd.DataFrame(sampled_data, columns=['x', 'y', 'z', 'target'])
    print("Testing Sampled DataFrame:")
    return sampled_df


    