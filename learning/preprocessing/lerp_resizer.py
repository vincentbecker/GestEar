"""Functions to linearly resample timeseries data.
"""

import numpy as np

EPSILON = 0.000001


def resize(imu_data, sampling_rate, start_time, end_time):
    if start_time < imu_data[0, 0]:
        imu_data = np.insert(imu_data, 0, np.concatenate(([start_time], imu_data[0, 1:])), axis=0)

    if imu_data[-1, 0] < end_time:
        imu_data = np.insert(imu_data, -1, np.concatenate(([end_time], imu_data[-1, 1:])), axis=0)

    dt = end_time - start_time
    l_data = len(imu_data)
    m = int(dt * sampling_rate)
    time_step = 1.0 / sampling_rate
    resized_data = np.zeros(shape=(m, 4))

    t = start_time
    current_index = 0
    for i in range(m):
        # Search the two timestamps around t and interpolate if necessary
        for j in range(current_index, l_data):
            entry = imu_data[j]
            timestamp = entry[0]
            if abs(timestamp - t) < EPSILON:
                # This sample has nearly the same timestamp
                resized_data[i, 0] = t
                resized_data[i, 1:] = entry[1:]
                current_index = j + 1
                break
            elif timestamp > t:
                # Search for entry with timestamp smaller than t
                previous_index = j - 1
                while imu_data[previous_index, 0] > t:
                    previous_index -= 1
                previous_entry = imu_data[previous_index]
                resized_data[i, 0] = t
                resized_data[i, 1:] = interpolate(t, previous_entry, entry)
                current_index = j
                break
        t += time_step
    return resized_data


def interpolate(t, previous_entry, entry):
    t1 = previous_entry[0]
    t2 = entry[0]
    dt = t2 - t1
    w1 = (t2 - t) / dt
    w2 = (t - t1) / dt
    return w1 * previous_entry[1:] + w2 * entry[1:]
