import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import public_data as pdata
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


# Create new Figure with black background
#创建figure窗口
fig, ax = plt.subplots(num='Intersection', figsize=(12, 12), facecolor='black')
rotate30_mat = np.array([[np.cos(np.pi/6), -1/2],[1/2, np.cos(np.pi/6)]])


def draw_coordinate_system():
    #设置坐标轴范围
    plt.xlim((-pdata.LANE_L, pdata.LANE_L))
    plt.ylim((-pdata.LANE_L, pdata.LANE_L))
    #设置坐标轴名称
    plt.xlabel(r'x(west - east)', color='white')
    plt.ylabel(r'y(south - north)', color='white')

    #设置坐标轴刻度
    my_x_ticks = np.arange(-pdata.LANE_L, pdata.LANE_L, pdata.LANE_W)
    my_y_ticks = np.arange(-pdata.LANE_L, pdata.LANE_L, pdata.LANE_W)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    # 设置坐标轴颜色
    plt.tick_params(colors='white')
    ax.set_facecolor('black')

# draw the edges of intersection
def draw_edges():
    #创建数据
    ed1 = [(-pdata.LANE_L, pdata.LANE_W),(-pdata.LANE_W, pdata.LANE_W)]
    ed2 = [(-pdata.LANE_W, pdata.LANE_W), (-pdata.LANE_W, pdata.LANE_L)]
    ed3 = [(pdata.LANE_W, pdata.LANE_L), (pdata.LANE_W, pdata.LANE_W)]
    ed4 = [(pdata.LANE_W, pdata.LANE_W), (pdata.LANE_L, pdata.LANE_W)]
    ed5 = [(pdata.LANE_L,-pdata.LANE_W), (pdata.LANE_W, -pdata.LANE_W)]
    ed6 = [(pdata.LANE_W, -pdata.LANE_W), (pdata.LANE_W, -pdata.LANE_L)]
    ed7 = [(-pdata.LANE_W, -pdata.LANE_L), (-pdata.LANE_W, -pdata.LANE_W)]
    ed8 = [(-pdata.LANE_W, -pdata.LANE_W), (-pdata.LANE_L, -pdata.LANE_W)]

    (ed1_x, ed1_y) = zip(*ed1)
    (ed2_x, ed2_y) = zip(*ed2)
    (ed3_x, ed3_y) = zip(*ed3)
    (ed4_x, ed4_y) = zip(*ed4)
    (ed5_x, ed5_y) = zip(*ed5)
    (ed6_x, ed6_y) = zip(*ed6)
    (ed7_x, ed7_y) = zip(*ed7)
    (ed8_x, ed8_y) = zip(*ed8)

    ax.add_line(Line2D(ed1_x, ed1_y, linewidth=1, color='white'))
    ax.add_line(Line2D(ed2_x, ed2_y, linewidth=1, color='white'))
    ax.add_line(Line2D(ed3_x, ed3_y, linewidth=1, color='white'))
    ax.add_line(Line2D(ed4_x, ed4_y, linewidth=1, color='white'))
    ax.add_line(Line2D(ed5_x, ed5_y, linewidth=1, color='white'))
    ax.add_line(Line2D(ed6_x, ed6_y, linewidth=1, color='white'))
    ax.add_line(Line2D(ed7_x, ed7_y, linewidth=1, color='white'))
    ax.add_line(Line2D(ed8_x, ed8_y, linewidth=1, color='white'))

    plt.plot()

def draw_separation():
    #创建数据
    sp1 = [(-pdata.LANE_L, 0),(-pdata.LANE_W, 0)]
    sp2 = [(0, pdata.LANE_W), (0, pdata.LANE_L)]
    sp3 = [(pdata.LANE_W, 0), (pdata.LANE_L, 0)]
    sp4 = [(0, -pdata.LANE_W), (0, -pdata.LANE_L)]

    (sp1_x, sp1_y) = zip(*sp1)
    (sp2_x, sp2_y) = zip(*sp2)
    (sp3_x, sp3_y) = zip(*sp3)
    (sp4_x, sp4_y) = zip(*sp4)

    ax.add_line(Line2D(sp1_x, sp1_y, linewidth=1, color='white', linestyle='--'))
    ax.add_line(Line2D(sp2_x, sp2_y, linewidth=1, color='white', linestyle='--'))
    ax.add_line(Line2D(sp3_x, sp3_y, linewidth=1, color='white', linestyle='--'))
    ax.add_line(Line2D(sp4_x, sp4_y, linewidth=1, color='white', linestyle='--'))

    plt.plot()


# 这种全盘清空重绘的方式性能如何不太清楚
# 每一帧都要调用环境中的 Agent 集合，以获取正确的四个点坐标
def draw_motorvehicle(i):
    plt.cla()   # clear object
    draw_coordinate_system()
    draw_edges()
    draw_separation()
    
    up_w = -pdata.LANE_W / 2 + pdata.MOTOR_W/2
    donw_w = up_w - pdata.MOTOR_W
    right_l = -pdata.LANE_W
    left_l = right_l - pdata.MOTER_L

    pgon = plt.Polygon([[right_l+i, up_w], [left_l+i, up_w], [left_l+i, donw_w], [right_l+i, donw_w]], color='g', alpha=1.0)
    ax.add_patch(pgon)
    # TODO: 之后再将车辆渲染更新的部分添加进来
    return ax  

def calculate_vertice(pos, vec):
        vertice = np.zeros((4, 2))

        _vertice_local = np.array([[pdata.MOTER_L/2, -pdata.MOTOR_W/2],
        [pdata.MOTER_L/2, pdata.MOTOR_W/2],
        [-pdata.MOTER_L/2, pdata.MOTOR_W/2],
        [-pdata.MOTER_L/2, -pdata.MOTOR_W/2]])

        _rotate90_mat = np.array([[0, -1],[1, 0]])
        world_x = vec / la.norm(vec)    
        world_y = np.matmul(_rotate90_mat, world_x)
        rotation_mat = np.array([[world_x[0], world_y[0]],[world_x[1], world_y[1]]])
        translation_vec = np.array([pos[0], pos[1]])
        for i in range(0, 4):
            tmp = np.matmul(rotation_mat, _vertice_local[i])
            vertice[i] = tmp + translation_vec
            
        return vertice

def draw():
    pos = np.array([0.0, 0.0])
    vec = np.array([2.0, 2.0])
    vertice = calculate_vertice(pos, vec)

    draw_coordinate_system()
    draw_edges()
    draw_separation()
    pgon = plt.Polygon(vertice, color='g', alpha=1.0)
    ax.add_patch(pgon)
    plt.show()

# test only
def render():    
    # 注：这里应该使动画持续到仿真结束
    # anim = FuncAnimation(fig, draw_motorvehicle, frames=np.arange(0, 24), interval=200)
    # plt.show()
    draw()


render()