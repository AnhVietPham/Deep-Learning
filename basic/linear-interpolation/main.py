import numpy as np
import math

if __name__ == "__main__":
    original_image = np.array([5, 4, 3, 2, 6])
    new_array = []
    length = len(original_image)
    new_length = 6
    ratio = (length - 1) / (new_length - 1)
    for i in range(new_length):
        x_floor_point = math.floor(ratio * i)
        x_ceil_point = math.ceil(ratio * i)
        y_floor_point = original_image[x_floor_point]
        y_ceil_point = original_image[x_ceil_point]
        print(f"Floor Point: {x_floor_point}, Ceil Point: {x_ceil_point}")
        print(f"Floor Value: {y_floor_point}, Ceil Value: {y_ceil_point}")
        if x_ceil_point == x_floor_point and y_floor_point == y_ceil_point:
            new_array.append(y_floor_point)
        else:
            xu, yu = x_ceil_point - x_floor_point, y_ceil_point - y_floor_point
            new_point = (((ratio * i) - x_floor_point) / xu) * yu + y_floor_point
            new_array.append(new_point)
    print(f'Original Array: {original_image}')
    print(f'New Array: {new_array}')
