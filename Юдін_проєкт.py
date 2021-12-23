import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def get_value(prompt, data_type):
    while True:
        value = input(prompt)
        try:
            value = data_type(value)
            return value
        except:
            print("Некоректний тип даних! Очікується:", data_type, '\n')


t_0 = get_value("Введіть t_0: ", float)
t_n = get_value("Введіть t_n: ", float)
h_t = get_value("Введіть крок h_t для змінної часу: ", float)
x_0 = get_value("Введіть x_0: ", float)
x_n = get_value("Введіть x_n: ", float)
h_x = get_value("Введіть крок h_x для змінної відстані: ", float)
tau = get_value("Введіть час реакції tau: ", float)
v_max = get_value("Введіть максимальну швидкість v_max: ", float)
k_max = get_value("Введіть максимальну густину k_max: ", float)
init_cond_file = get_value("Введіть назву файлу з початковими умовами для v та k: ", str)

"""
# діапазон значень незалежних змінних, інші параметри
t_bounds = np.array([0, 120])
t_0, t_n = t_bounds
x_bounds = np.array([0, 300])
x_0, x_n = x_bounds
tau = 2.5
v_max = 10
k_max = 1
#h = 20
# крок, з яким шукаються значеня функцій v та k
h_t = 0.1
h_x = 2
init_cond_file = 'init_cond_3.csv'
"""



# функція оптимальної швидкості
def v_opt(i, j):
    return v_max * (1 - k[i, j] / k_max)


# матриці для збереження значень функцій v та k
v = np.zeros((round((t_n - t_0) / h_t) + 1,
              round((x_n - x_0) / h_x) + 1))
k = np.zeros_like(v)


"""
# початкові умови
#k[0, :] = [0.2 if x < 100 else 0.01 for x in np.arange(x_0, x_n+h_x, h_x)]
#k[0, :] = [0.15 if x <= 130 else (0.1 if x >= 280 else 0.8) for x in np.arange(x_0, x_n+h_x, h_x)]
k[0, :] = [0.5*np.sin(0.05*x)+0.5 for x in np.arange(x_0, x_n+h_x, h_x)]
k[0, k.shape[1] - 1] = k[0, 0]

v[0, :] = [v_max * (1 - k[0, j] / k_max) for j in range(v.shape[1])]
#print(v.shape)"""


#aa = np.stack((v[0], k[0]))
#print(aa)
#np.savetxt(init_cond_file, aa, delimiter = ',', header = 'v, k')


init_cond = np.genfromtxt(init_cond_file, delimiter = ';')
v[0] = init_cond[0]
k[0] = init_cond[1]

#print(np.min(v[0]))
#print(np.max(v[0]))



# ==================================================================================



print("Обчислюємо...")
# сам процес розв'язання системи дифрівнянь
for i in range(v.shape[0] - 1):
    #print(i)
    for j in range(1, v.shape[1]):

        k[i+1, j] = np.clip(
            k[i, j] + h_t / h_x * (k[i, j-1] * v[i, j-1] - k[i, j] * v[i, j]),
            np.min(k[0]), np.max(k[0]))

        v[i+1, j] = np.clip(
            v[i, j] + h_t / tau * (v_opt(i, j) - v[i, j]) + h_t / h_x * v[i, j] * (
            v[i, j-1] - v[i, j]) - 20 * h_t / (tau * h_x) * (k[i, j+1 if j != v.shape[1]-1 else 1] - k[i, j]) / 
            (k[i, j] + 0.1),
            np.min(v[0]), np.max(v[0]))


    # врахування циклічності дороги
    k[i+1, 0] = k[i+1, v.shape[1] - 1]
    v[i+1, 0] = v[i+1, v.shape[1] - 1]



# ==================================================================================


"""
# зберегти матриці в csv-файлі
np.savetxt("result_v_3.csv", v, delimiter = ',')
np.savetxt("result_k_3.csv", k, delimiter = ',')
"""

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9, 6))
fig.tight_layout(pad = 4)

# intialize two line objects (one in each axes)
line_v, = ax1.plot([], [], lw=3)
line_k, = ax2.plot([], [], lw=3, color='r')
title = ax1.text(0, np.max(v) + (v_max-np.min(v[0])) / 6, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                )
variables = [line_v, line_k, title]

ax1.set_title("Швидкість потоку")
ax1.set_xlim(0, x_n)
ax1.set_ylim(np.min(v), np.max(v))
ax1.set_ylabel("v")
ax2.set_title("Густина потоку")
ax2.set_xlim(0, x_n)
ax2.set_ylim(np.min(k), k_max)
ax2.set_xlabel("х")
ax2.set_ylabel("k")

# статичні графіки
"""x = np.linspace(x_0, x_n, round((x_n - x_0) / h_x + 1))
v_sec_1, = ax1.plot(x, v[10], "r-", label = "1 с")
v_sec_10, = ax1.plot(x, v[100], "g-", label = "10 с")
v_sec_30, = ax1.plot(x, v[300], "b-", label = "30 с")
v_sec_60, = ax1.plot(x, v[600], "y-", label = "60 с")
v_sec_120, = ax1.plot(x, v[1200], "m-", label = "120 с")
plt.legend(handles = [v_sec_1, v_sec_10, v_sec_30, v_sec_60, v_sec_120], bbox_to_anchor=(1.04,1), borderaxespad=0)
k_sec_1, = ax2.plot(x, k[10], "r-", label = "1 с")
k_sec_10, = ax2.plot(x, k[100], "g-", label = "10 с")
k_sec_30, = ax2.plot(x, k[300], "b-", label = "30 с")
k_sec_60, = ax2.plot(x, k[600], "y-", label = "60 с")
k_sec_120, = ax2.plot(x, k[1200], "m-", label = "120 с")
ax2.legend(handles = [k_sec_1, k_sec_10, k_sec_30, k_sec_60, k_sec_120])
plt.show()"""


def data_gen():
    i = 0
    x = np.linspace(x_0, x_n, round((x_n - x_0) / h_x + 1))
    t_arr = np.linspace(t_0, t_n, round((t_n - t_0) / h_t + 1))
    while i < v.shape[0]:
        y_v = v[i]
        y_k = k[i]
        t = t_arr[i]
        i += 1
        yield t, x, y_v, y_k


def run(data):
    t, x, y_v, y_k = data
    variables[0].set_data(x, y_v)
    variables[1].set_data(x, y_k)
    variables[2].set_text(f"Стан транспортного потоку, t = {round(t, 1)}")
    return variables


anim = FuncAnimation(fig, run, data_gen, interval=30, blit=True, save_count=v.shape[0])
#writervideo = animation.FFMpegWriter(fps=20) 
#anim.save("traffic_flow_anim_3.mp4", writer=writervideo)

plt.show()