import matplotlib.pyplot as plt

# 假设有一个二维列表作为表格数据
table_data = [
    ["Name", "Age", "City"],
    ["Alice", 25, "New York"],
    ["Bob", 30, "San Francisco"],
    ["Charlie", 22, "Los Angeles"]
]

# 创建一个新的图形
fig, ax = plt.subplots()

# 隐藏坐标轴
ax.axis('off')

# 创建表格
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLabels=None, cellColours=None)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)

# 调整表格布局
table.auto_set_column_width([0, 1, 2])

plt.show()
