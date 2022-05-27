import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


# ready
def otkl_a(data):
    empty = []
    for i in range(0, len(data), 10):
        array = 1.0 * np.array(data[i:i + 10])
        n = len(array)
        m, se = np.mean(array), st.sem(array)
        h = se * st.t.ppf((1 + 0.95) / 2., n - 1)
        ind = []
        for i in range(0, len(array)):
            if (array[i] > m + h or array[i] < m - h):
                ind.append(i)
        array = np.delete(array, ind)
        empty.append(np.mean(array))
    return delnans(empty)


# ready
def otkl(array):
    array = 1.0 * np.array(array)
    n = len(array)
    m, se = np.mean(array), st.sem(array)
    h = se * st.t.ppf((1 + 0.95) / 2., n - 1)
    ind = []
    for i in range(0, len(array)):
        if (array[i] > m + h or array[i] < m - h):
            ind.append(i)
    array = np.delete(array, ind)
    return np.mean(array)


# no need, but ready
def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1)


# no need, but ready
def ticks_minimal1(array, h):
    new_array = []
    i = 0
    k = 0
    while (len(array) > i + 1):
        if (array[i][0] + h * k <= array[i + 1][0]):
            new_array.append(array[i][1])
            k += 1
        else:
            k = 0
            i += 1
    return new_array


# ready
def ticks_minimal(array, h):
    new_array = []
    i = 0
    k = 0
    while (500000 > int(h) * i and k < len(array)):
        if (int(h) * i < array[k][0]):
            new_array.append(array[k][1])
            i += 1
        else:
            new_array.append(array[k][1])
            k += 1
            i += 1
    return new_array


# no need, but ready
def min_inter(array):
    minint = 20000
    for row in range(0, len(array) - 1, 1):
        if (minint > (array[row + 1][0] - array[row][0]) and (array[row + 1][0] - array[row][0]) != 0):
            minint = array[row + 1][0] - array[row][0]

    print('Минимальный шаг в наборе: ' + str(int(minint)))
    return int(minint)


def delnans(array):
    if np.isnan(array[0]):
        array[0] = 0
    for i in range(1, len(array)):
        if (np.isnan(array[i])):
            array[i] = array[i - 1]
    return array


# ready
def ticks_in_frames(tick_array, tpf):
    frame_array = []
    h = tpf
    tmp = []
    for i in range(0, len(tick_array)):
        if (i % h == 0):
            if (i == 0):
                tmp = 0
                frame_array.append(tmp)
                tmp = []
                i += 1
                continue
            else:
                frame_array.append(otkl(tmp))
                tmp = []
                i += 1
        if (np.isnan(tick_array[i])):
            tmp.append(0)
        else:
            tmp.append(tick_array[i])
    frame_array = delnans(frame_array)
    if (len(frame_array) > 1250):
        return frame_array[0:1250]
    else:
        ze = np.zeros(1250 - np.size(frame_array))
        ze[:] = frame_array[-1]
        return np.append(frame_array, ze)


# ready
def reader(path):
    with open(path, encoding='utf-8') as r_file:
        array = []
        reader = csv.reader(r_file, delimiter=";")
        for row in reader:
            array.append([int(row[0]), float(row[1])])
        return array


def interval_true(data):
    result = []
    for i in range(0, len(data[0]), 1):
        # print([data[0][i],data[1][i],data[2][i],data[3][i],data[4][i]])
        array = 1.0 * np.array([data[0][i], data[1][i], data[2][i], data[3][i], data[4][i]])
        where_are_NaNs = np.isnan(array)
        array[where_are_NaNs] = 0
        # print(array)
        n = len(array)
        m, se = np.mean(array), st.sem(array)
        h = se * st.t.ppf((1 + 0.95) / 2., n - 1)
        m1h = 0 if m - h < 0 else m - h
        m2h = 1 if m + h > 1 else m + h
        # print('1   _'+str(m) + ' ' + str(m-h) + ' ' + str(m+h))
        ind = []
        for j in range(0, len(array)):
            if (array[j] > m2h or array[j] < m1h):
                ind.append(j)
        # print(ind)
        # print(array)
        array = np.delete(array, ind)
        # print(array)
        # input("Press Enter to continue...")
        m, se = np.mean(array), st.sem(array)
        h = se * st.t.ppf((1 + 0.95) / 2., n - 1)
        # print('2   _'+str(m) + ' ' + str(m-h) + ' ' + str(m+h))
        m1h = 0 if m - h < 0 else m - h
        m2h = 1 if m + h > 1 else m + h
        result = np.append(result, [m, m1h, m2h, h])
    return result


def dis1():
    ######################################################################################################################
    # читаем данные 1ой позиции
    subject1_poz1_tics = np.array(reader(path[0][0]))
    subject2_poz1_tics = np.array(reader(path[0][1]))
    subject3_poz1_tics = np.array(reader(path[0][2]))
    subject4_poz1_tics = np.array(reader(path[0][3]))
    subject5_poz1_tics = np.array(reader(path[0][4]))
    subject6_poz1_tics = np.array(reader(path[0][5]))
    subject7_poz1_tics = np.array(reader(path[0][6]))

    # массив граффиков 1ой позиции: сухой
    list_poz1_tics = [subject1_poz1_tics,
                      subject2_poz1_tics,
                      subject3_poz1_tics,
                      subject4_poz1_tics,
                      subject5_poz1_tics,
                      subject6_poz1_tics,
                      subject7_poz1_tics
                      ]

    # шаг для прохода
    h = 100

    # массив граффиков 1 позиции с шагом по 100
    list_ticks_minimal = [ticks_minimal(list_poz1_tics[0], h),
                          ticks_minimal(list_poz1_tics[1], h),
                          ticks_minimal(list_poz1_tics[2], h),
                          ticks_minimal(list_poz1_tics[3], h),
                          ticks_minimal(list_poz1_tics[4], h),
                          ticks_minimal(list_poz1_tics[5], h),
                          ticks_minimal(list_poz1_tics[6], h)
                          ]

    # массив колличества отсчетов на кадр
    tpf = [math.floor(len(list_ticks_minimal[0]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[1]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[2]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[3]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[4]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[5]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[6]) / (seconds * framerate))
           ]

    # разделение на кадры и удаление выбросов в исходных данных
    subject1_poz1_frames = ticks_in_frames(list_ticks_minimal[0], tpf[0])
    subject2_poz1_frames = ticks_in_frames(list_ticks_minimal[1], tpf[1])
    subject3_poz1_frames = ticks_in_frames(list_ticks_minimal[2], tpf[2])
    subject4_poz1_frames = ticks_in_frames(list_ticks_minimal[3], tpf[3])
    subject5_poz1_frames = ticks_in_frames(list_ticks_minimal[4], tpf[5])
    subject6_poz1_frames = ticks_in_frames(list_ticks_minimal[5], tpf[5])
    subject7_poz1_frames = ticks_in_frames(list_ticks_minimal[6], tpf[6])

    # массив кадров
    list_poz1_frames = [subject1_poz1_frames,
                        subject2_poz1_frames,
                        subject3_poz1_frames,
                        subject4_poz1_frames,
                        subject5_poz1_frames,
                        subject6_poz1_frames,
                        subject7_poz1_frames
                        ]

    correlation_value = [st.pearsonr(np.arange(0, 1250), list_poz1_frames[0])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[1])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[2])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[3])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[4])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[5])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[6])[0]
                         ]

    legend_test = [f'Values of the first test subject',
                   f'Values of the second test subject',
                   f'Values of the third test subject',
                   f'Values of the fourth test subject',
                   f'Values of the fifth test subject',
                   f'Values of the sixth test subject',
                   f'Values of the seventh test subject',
                   f'Values of the eighth test subject'
                   ]

    correlation_string = f'Correlation value [1] = {correlation_value[0]}\nCorrelation value [2] = {correlation_value[1]}\nCorrelation value [3] = {correlation_value[2]}\nCorrelation value [4] = {correlation_value[3]}\nCorrelation value [5] = {correlation_value[4]}\nCorrelation value [6] = {correlation_value[5]}\nCorrelation value [7] = {correlation_value[6]}'
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    # построение 5 эксперементов на одном графике
    plt.plot(list_poz1_frames[0], colors[0])
    plt.plot(list_poz1_frames[1], colors[1])
    plt.plot(list_poz1_frames[2], colors[2])
    plt.plot(list_poz1_frames[3], colors[3])
    plt.plot(list_poz1_frames[4], colors[4])
    plt.plot(list_poz1_frames[5], color='black')
    plt.plot(list_poz1_frames[6], color='blue')
    plt.xlim(d_range)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_lable)
    plt.title(position_name[0])
    plt.text(0, 0, correlation_string, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
    plt.legend(legend_test, loc='lower center')
    plt.grid()
    plt.savefig('C:\\Users\\chern\\OneDrive\\Рабочий стол\\resultimages\\poz1_7graph.png', dpi=250)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    result = interval_true(list_poz1_frames)
    plt.plot(result[0::4])
    plt.plot(result[1::4], "--", color="green")
    plt.plot(result[2::4], "--", color="green")
    print(len(result[0::4]))
    x = st.pearsonr(np.arange(0, 1250), result[0::4])[0]
    str = f'Average standard deviation =  {np.mean(result[4::4])}\nCorrelation value = {x}'
    plt.xlim(d_range)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_lable)

    # Turn off tick labels
    plt.title('First position in experiment: averaged values')
    plt.text(0, 0, str, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
    plt.legend(['middle value of the confidence interval', 'lower value of the confidence interval',
                'upper value of the confidence interval'], loc='lower center')
    plt.grid()
    plt.savefig('C:\\Users\\chern\\OneDrive\\Рабочий стол\\resultimages\\poz1_result.png', dpi=250)
    # plt.show()
    return result[0::4]


def dis2():
    ######################################################################################################################
    # читаем данные 1ой позиции
    subject1_poz1_tics = np.array(reader(path[1][0]))
    subject2_poz1_tics = np.array(reader(path[1][1]))
    subject3_poz1_tics = np.array(reader(path[1][2]))
    subject4_poz1_tics = np.array(reader(path[1][3]))
    subject5_poz1_tics = np.array(reader(path[1][4]))
    subject6_poz1_tics = np.array(reader(path[1][5]))
    subject7_poz1_tics = np.array(reader(path[1][6]))
    subject8_poz1_tics = np.array(reader(path[1][7]))
    # массив граффиков 1ой позиции: сухой
    list_poz1_tics = [subject1_poz1_tics,
                      subject2_poz1_tics,
                      subject3_poz1_tics,
                      subject4_poz1_tics,
                      subject5_poz1_tics,
                      subject6_poz1_tics,
                      subject7_poz1_tics,
                      subject8_poz1_tics,
                      ]

    # шаг для прохода
    h = 10

    # массив граффиков 1 позиции с шагом по 100
    list_ticks_minimal = [ticks_minimal(list_poz1_tics[0], h),
                          ticks_minimal(list_poz1_tics[1], h),
                          ticks_minimal(list_poz1_tics[2], h),
                          ticks_minimal(list_poz1_tics[3], h),
                          ticks_minimal(list_poz1_tics[4], h),
                          ticks_minimal(list_poz1_tics[5], h),
                          ticks_minimal(list_poz1_tics[6], h),
                          ticks_minimal(list_poz1_tics[7], h)
                          ]

    # массив колличества отсчетов на кадр
    tpf = [math.floor(len(list_ticks_minimal[0]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[1]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[2]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[3]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[4]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[5]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[6]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[7]) / (seconds * framerate))
           ]

    # разделение на кадры и удаление выбросов в исходных данных
    subject1_poz1_frames = ticks_in_frames(list_ticks_minimal[0], tpf[0])
    subject2_poz1_frames = ticks_in_frames(list_ticks_minimal[1], tpf[1])
    subject3_poz1_frames = ticks_in_frames(list_ticks_minimal[2], tpf[2])
    subject4_poz1_frames = ticks_in_frames(list_ticks_minimal[3], tpf[3])
    subject5_poz1_frames = ticks_in_frames(list_ticks_minimal[4], tpf[4])
    subject6_poz1_frames = ticks_in_frames(list_ticks_minimal[4], tpf[4])
    subject7_poz1_frames = ticks_in_frames(list_ticks_minimal[6], tpf[6])
    subject8_poz1_frames = ticks_in_frames(list_ticks_minimal[7], tpf[7])

    # массив кадров
    list_poz1_frames = [subject1_poz1_frames,
                        subject2_poz1_frames,
                        subject3_poz1_frames,
                        subject4_poz1_frames,
                        subject5_poz1_frames,
                        subject6_poz1_frames,
                        subject7_poz1_frames,
                        subject8_poz1_frames,
                        ]

    correlation_value = [st.pearsonr(np.arange(0, 1250), list_poz1_frames[0])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[1])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[2])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[3])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[4])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[5])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[6])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[7])[0]
                         ]

    legend_test = [f'Values of the first test subject',
                   f'Values of the second test subject',
                   f'Values of the third test subject',
                   f'Values of the fourth test subject',
                   f'Values of the fifth test subject',
                   f'Values of the sixth test subject',
                   f'Values of the seventh test subject',
                   f'Values of the eighth test subject'
                   ]

    correlation_string = f'Correlation value [1] = {correlation_value[0]}\nCorrelation value [2] = {correlation_value[1]}\nCorrelation value [3] = {correlation_value[2]}\nCorrelation value [4] = {correlation_value[3]}\nCorrelation value [5] = {correlation_value[4]}\nCorrelation value [6] = {correlation_value[5]}\nCorrelation value [7] = {correlation_value[6]}\nCorrelation value [8] = {correlation_value[7]}'
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    # построение 5 эксперементов на одном графике
    plt.plot(list_poz1_frames[0], colors[0])
    plt.plot(list_poz1_frames[1], colors[1])
    plt.plot(list_poz1_frames[2], colors[2])
    plt.plot(list_poz1_frames[3], colors[3])
    plt.plot(list_poz1_frames[4], colors[4])
    plt.plot(list_poz1_frames[5], color='black')
    plt.plot(list_poz1_frames[6], color='blue')
    plt.plot(list_poz1_frames[7], color='cyan')
    plt.xlim(d_range)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_lable)
    plt.title(position_name[1])
    plt.text(0, 0, correlation_string, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
    plt.legend(legend_test, loc='lower center')
    plt.grid()
    plt.savefig('C:\\Users\\chern\\OneDrive\\Рабочий стол\\resultimages\\poz2_8graph.png', dpi=250)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    result = interval_true(list_poz1_frames)
    plt.plot(result[0::4])
    plt.plot(result[1::4], "--", color="green")
    plt.plot(result[2::4], "--", color="green")
    print(len(result[0::4]))
    x = st.pearsonr(np.arange(0, 1250), result[0::4])[0]
    str = f'Average standard deviation =  {np.mean(result[4::4])}\nCorrelation value = {x}'
    plt.xlim(d_range)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_lable)

    # Turn off tick labels
    plt.title('Second position in experiment: averaged values')
    plt.text(0, 0, str, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
    plt.legend(['middle value of the confidence interval', 'lower value of the confidence interval',
                'upper value of the confidence interval'], loc='lower center')
    plt.grid()
    plt.savefig('C:\\Users\\chern\\OneDrive\\Рабочий стол\\resultimages\\poz2_result.png', dpi=250)
    # plt.show()
    return result[0::4]


def dis3():
    ######################################################################################################################
    # читаем данные 1ой позиции
    subject1_poz1_tics = np.array(reader(path[2][0]))
    subject2_poz1_tics = np.array(reader(path[2][1]))
    subject3_poz1_tics = np.array(reader(path[2][2]))
    subject4_poz1_tics = np.array(reader(path[2][3]))
    subject5_poz1_tics = np.array(reader(path[2][4]))
    subject6_poz1_tics = np.array(reader(path[2][5]))
    subject7_poz1_tics = np.array(reader(path[2][6]))
    subject8_poz1_tics = np.array(reader(path[2][7]))
    subject9_poz1_tics = np.array(reader(path[2][8]))


    # массив граффиков 1ой позиции: сухой
    list_poz1_tics = [subject1_poz1_tics,
                      subject2_poz1_tics,
                      subject3_poz1_tics,
                      subject4_poz1_tics,
                      subject5_poz1_tics,
                      subject6_poz1_tics,
                      subject7_poz1_tics,
                      subject8_poz1_tics,
                      subject9_poz1_tics
                      ]

    # шаг для прохода
    h = 10

    # массив граффиков 1 позиции с шагом по 100
    list_ticks_minimal = [ticks_minimal(list_poz1_tics[0], h),
                          ticks_minimal(list_poz1_tics[1], h),
                          ticks_minimal(list_poz1_tics[2], h),
                          ticks_minimal(list_poz1_tics[3], h),
                          ticks_minimal(list_poz1_tics[4], h),
                          ticks_minimal(list_poz1_tics[5], h),
                          ticks_minimal(list_poz1_tics[6], h),
                          ticks_minimal(list_poz1_tics[7], h),
                          ticks_minimal(list_poz1_tics[8], h)
                          ]

    # массив колличества отсчетов на кадр
    tpf = [math.floor(len(list_ticks_minimal[0]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[1]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[2]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[3]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[4]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[5]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[6]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[7]) / (seconds * framerate)),
           math.floor(len(list_ticks_minimal[8]) / (seconds * framerate))
           ]

    # разделение на кадры и удаление выбросов в исходных данных
    subject1_poz1_frames = ticks_in_frames(list_ticks_minimal[0], tpf[0])
    subject2_poz1_frames = ticks_in_frames(list_ticks_minimal[1], tpf[1])
    subject3_poz1_frames = ticks_in_frames(list_ticks_minimal[2], tpf[2])
    subject4_poz1_frames = ticks_in_frames(list_ticks_minimal[3], tpf[3])
    subject5_poz1_frames = ticks_in_frames(list_ticks_minimal[4], tpf[4])
    subject6_poz1_frames = ticks_in_frames(list_ticks_minimal[5], tpf[5])
    subject7_poz1_frames = ticks_in_frames(list_ticks_minimal[6], tpf[6])
    subject8_poz1_frames = ticks_in_frames(list_ticks_minimal[7], tpf[7])
    subject9_poz1_frames = ticks_in_frames(list_ticks_minimal[8], tpf[8])

    # массив кадров
    list_poz1_frames = [subject1_poz1_frames,
                        subject2_poz1_frames,
                        subject3_poz1_frames,
                        subject4_poz1_frames,
                        subject5_poz1_frames,
                        subject6_poz1_frames,
                        subject7_poz1_frames,
                        subject8_poz1_frames,
                        subject9_poz1_frames
                        ]

    correlation_value = [st.pearsonr(np.arange(0, 1250), list_poz1_frames[0])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[1])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[2])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[3])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[4])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[5])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[6])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[7])[0],
                         st.pearsonr(np.arange(0, 1250), list_poz1_frames[8])[0]
                         ]

    legend_test = [f'Values of the first test subject',
                   f'Values of the second test subject',
                   f'Values of the third test subject',
                   f'Values of the fourth test subject',
                   f'Values of the fifth test subject',
                   f'Values of the sixth test subject',
                   f'Values of the seventh test subject',
                   f'Values of the eighth test subject',
                   f'Values of the ninth test subject'
                   ]

    correlation_string = f'Correlation value [1] = {correlation_value[0]}\nCorrelation value [2] = {correlation_value[1]}\nCorrelation value [3] = {correlation_value[2]}\nCorrelation value [4] = {correlation_value[3]}\nCorrelation value [5] = {correlation_value[4]}\nCorrelation value [6] = {correlation_value[5]}\nCorrelation value [7] = {correlation_value[6]}\nCorrelation value [8] = {correlation_value[7]}\nCorrelation value [9] = {correlation_value[8]}'
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    # построение 9 эксперементов на одном графике
    plt.plot(list_poz1_frames[0], colors[0])
    plt.plot(list_poz1_frames[1], colors[1])
    plt.plot(list_poz1_frames[2], colors[2])
    plt.plot(list_poz1_frames[3], colors[3])
    plt.plot(list_poz1_frames[4], colors[4])
    plt.plot(list_poz1_frames[5], color='black')
    plt.plot(list_poz1_frames[6], color='blue')
    plt.plot(list_poz1_frames[7], color='cyan')
    plt.plot(list_poz1_frames[8], color='grey')
    plt.xlim(d_range)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_lable)
    plt.title(position_name[2])
    plt.text(0, 0, correlation_string, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
    plt.legend(legend_test, loc='lower center')
    plt.grid()
    plt.savefig('C:\\Users\\chern\\OneDrive\\Рабочий стол\\resultimages\\poz3_9graph.png', dpi=250)
    # plt.show()

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    result = interval_true(list_poz1_frames)
    plt.plot(result[0::4])
    plt.plot(result[1::4], "--", color="green")
    plt.plot(result[2::4], "--", color="green")
    print(len(result[0::4]))
    x = st.pearsonr(np.arange(0, 1250), result[0::4])[0]
    str = f'Average standard deviation =  {np.mean(result[4::4])}\nCorrelation value = {x}'
    plt.xlim(d_range)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_lable)

    # Turn off tick labels
    plt.title('Third position in experiment: averaged values')
    plt.text(0, 0, str, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
    plt.legend(['middle value of the confidence interval', 'lower value of the confidence interval',
                'upper value of the confidence interval'], loc='lower center')
    plt.grid()
    plt.savefig('C:\\Users\\chern\\OneDrive\\Рабочий стол\\resultimages\\poz3_result.png', dpi=250)
    # plt.show()
    return result[0::4]



y_label = 'Metric value'
x_lable = 'Video frame index'

seconds = 50  # длина видео
framerate = 25  # кадров в секунду

start_frame = 0
end_frame = seconds * framerate
d_range = [start_frame, end_frame]

# массив цветов
colors = ['orange', 'g', 'r', 'y', 'm']

# пройдемся по всем кадрам, апроксимируем значения для каждого кадра в диапозоне от 0:tpf:ticks // получится, что пройдем по кадрам
# ага, надейся, по 100 отсчетов мы идем, молодой ещё, наивный
position_name = ['First position', 'Second position', 'Third position']

# первое индекс - позиция // второй индекс - испытуемый \\ ссылки на .csv
path = [
    ['C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test00.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test01.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test02.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test03.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test04.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test05.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test06.csv'],
    #########################################
    ['C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test10.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test11.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test12.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test13.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test14.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test15.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test16.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test17.csv'],
    #########################################
    ['C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test20.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test21.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test22.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test23.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test24.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test25.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test26.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test27.csv',
     'C:\\Users\\chern\\OneDrive\\Рабочий стол\\bigdata\\test28.csv']
]

arr = [dis1(), dis2(), dis3()]

correlation_value = [st.pearsonr(np.arange(0, 1250), arr[0])[0],
                     st.pearsonr(np.arange(0, 1250), arr[1])[0],
                     st.pearsonr(np.arange(0, 1250), arr[2])[0]]

legend_test = [f'Averaged values of the first position',
               f'Averaged values of the second position',
               f'Averaged values of the third position'
               ]

list_avg = [np.mean(arr[0]),
            np.mean(arr[1]),
            np.mean(arr[2]),
            ]

correlation_string = f'Correlation value [1] = {correlation_value[0]}\nCorrelation value [2] = {correlation_value[1]}\nCorrelation value [3] = {correlation_value[2]}'
avg_string = f'AVG value [1] = {list_avg[0]}\nAVG value [2] = {list_avg[1]}\nAVG value [3] = {list_avg[2]}\n'
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(15, 7))
# построение 5 эксперементов на одном графике
plt.plot(arr[0], colors[0])
plt.plot(arr[1], colors[1])
plt.plot(arr[2], colors[2])
plt.xlim(d_range)
ax.set_ylabel(y_label)
ax.set_xlabel(x_lable)
plt.title('Resulting graphs for first, second and third positions')
plt.text(0, 0, correlation_string, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
plt.text(950, 0, avg_string, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
plt.legend(legend_test, loc='lower center')
plt.grid()
plt.savefig('C:\\Users\\chern\\OneDrive\\Рабочий стол\\resultimages\\result.png', dpi=250)
# plt.show()

list_10kad = [otkl_a(arr[0]),
              otkl_a(arr[1]),
              otkl_a(arr[2])]

correlation_value_10kad = [st.pearsonr(np.arange(0, 125), list_10kad[0])[0],
                           st.pearsonr(np.arange(0, 125), list_10kad[1])[0],
                           st.pearsonr(np.arange(0, 125), list_10kad[2])[0]]

list_avg = [np.mean(list_10kad[0]),
            np.mean(list_10kad[1]),
            np.mean(list_10kad[2]),
            ]

correlation_string = f'Correlation value [1] = {correlation_value_10kad[0]}\nCorrelation value [2] = {correlation_value_10kad[1]}\nCorrelation value [3] = {correlation_value_10kad[2]}'
avg_string = f'AVG value [1] = {list_avg[0]}\nAVG value [2] = {list_avg[1]}\nAVG value [3] = {list_avg[2]}'
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(15, 7))
# построение 5 эксперементов на одном графике
plt.plot(list_10kad[0], colors[0])
plt.plot(list_10kad[1], colors[1])
plt.plot(list_10kad[2], colors[2])
plt.xlim([0, 125])
ax.set_ylabel(y_label)
ax.set_xlabel('Video tick index (1 tick = 10 frames)')
plt.title('Resulting graphs for first, second and third position')
plt.text(0, 0, correlation_string, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
plt.text(95, 0, avg_string, fontsize=10, bbox={'facecolor': 'yellow', 'alpha': 0.2})
plt.legend(legend_test, loc='lower center')
plt.grid()
plt.savefig('C:\\Users\\chern\\OneDrive\\Рабочий стол\\resultimages\\result_10frames.png', dpi=250)

