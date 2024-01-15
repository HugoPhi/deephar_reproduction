import matplotlib.pyplot as plt
import numpy as np

confusion_matrix = np.random.randint(0, 100, size=(7, 7))

matrix_height, matrix_width = confusion_matrix.shape

fig, ax = plt.subplots(figsize=(matrix_width + 1, matrix_height))

cax = ax.matshow(confusion_matrix, cmap='Greens')

fig.colorbar(cax, fraction=0.046, pad=0.04)

for i in range(matrix_height):
    for j in range(matrix_width):
        ax.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')

ax.set_xticks(np.arange(matrix_width))
ax.set_yticks(np.arange(matrix_height))
ax.set_xticklabels(['Class {}'.format(i) for i in range(1, matrix_width + 1)], ha='left')
ax.set_yticklabels(['Class {}'.format(i) for i in range(1, matrix_height + 1)])

# 添加标签
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 最大化显示窗口
plt.get_current_fig_manager().window.state('withdrawn')

# 保存图形
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

plt.show()
