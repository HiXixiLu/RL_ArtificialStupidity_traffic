import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D


# Create new Figure with black background
#创建figure窗口
fig, ax = plt.subplots(num="Intersection", figsize=(12, 12), facecolor="black")


def draw_coordinate_system():
    #设置坐标轴范围
    plt.xlim((-20, 20))
    plt.ylim((-20, 20))
    #设置坐标轴名称
    plt.xlabel(r'x(west - east)', color="white")
    plt.ylabel(r'y(south - north)', color="white")

    #设置坐标轴刻度
    my_x_ticks = np.arange(-20, 20, 2)
    my_y_ticks = np.arange(-20, 20, 2)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    # 设置坐标轴颜色
    plt.tick_params(colors="white")
    ax.set_facecolor("black")

# draw the edges of intersection
def draw_edges():
    #创建数据
    ed1 = [(-20, 2),(-2, 2)]
    ed2 = [(-2, 2), (-2, 20)]
    ed3 = [(2, 20), (2, 2)]
    ed4 = [(2, 2), (20, 2)]
    ed5 = [(20,-2), (2, -2)]
    ed6 = [(2, -2), (2, -20)]
    ed7 = [(-2, -20), (-2, -2)]
    ed8 = [(-2, -2), (-20, -2)]

    (ed1_x, ed1_y) = zip(*ed1)
    (ed2_x, ed2_y) = zip(*ed2)
    (ed3_x, ed3_y) = zip(*ed3)
    (ed4_x, ed4_y) = zip(*ed4)
    (ed5_x, ed5_y) = zip(*ed5)
    (ed6_x, ed6_y) = zip(*ed6)
    (ed7_x, ed7_y) = zip(*ed7)
    (ed8_x, ed8_y) = zip(*ed8)

    ax.add_line(Line2D(ed1_x, ed1_y, linewidth=1, color="white"))
    ax.add_line(Line2D(ed2_x, ed2_y, linewidth=1, color="white"))
    ax.add_line(Line2D(ed3_x, ed3_y, linewidth=1, color="white"))
    ax.add_line(Line2D(ed4_x, ed4_y, linewidth=1, color="white"))
    ax.add_line(Line2D(ed5_x, ed5_y, linewidth=1, color="white"))
    ax.add_line(Line2D(ed6_x, ed6_y, linewidth=1, color="white"))
    ax.add_line(Line2D(ed7_x, ed7_y, linewidth=1, color="white"))
    ax.add_line(Line2D(ed8_x, ed8_y, linewidth=1, color="white"))

    plt.plot()

def draw_separation():
    #创建数据
    sp1 = [(-20, 0),(-2, 0)]
    sp2 = [(0, 2), (0, 20)]
    sp3 = [(2, 0), (20, 0)]
    sp4 = [(0, -2), (0, -20)]

    (sp1_x, sp1_y) = zip(*sp1)
    (sp2_x, sp2_y) = zip(*sp2)
    (sp3_x, sp3_y) = zip(*sp3)
    (sp4_x, sp4_y) = zip(*sp4)

    ax.add_line(Line2D(sp1_x, sp1_y, linewidth=1, color="white", linestyle="--"))
    ax.add_line(Line2D(sp2_x, sp2_y, linewidth=1, color="white", linestyle="--"))
    ax.add_line(Line2D(sp3_x, sp3_y, linewidth=1, color="white", linestyle="--"))
    ax.add_line(Line2D(sp4_x, sp4_y, linewidth=1, color="white", linestyle="--"))

    plt.plot()

draw_coordinate_system()
draw_edges()
draw_separation()
plt.show()
