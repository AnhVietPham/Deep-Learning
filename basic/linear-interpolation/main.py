import numpy as np
import math

if __name__ == "__main__":
    original_image = np.array([5, 4, 3, 2, 6])
    new_array = []
    length = len(original_image)
    new_length = 6
    ratio = (length - 1) / (new_length - 1)
    for i in range(new_length):
        floor_point = math.floor(ratio * i)
        ceil_point = math.ceil(ratio * i)
        print(f"Floor Point: {floor_point}, Ceil Point: {ceil_point}")
        print(f"Floor Value: {original_image[floor_point]}, Ceil Value: {original_image[ceil_point]}")
        if ceil_point == floor_point and original_image[floor_point] == original_image[ceil_point]:
            new_array.append(original_image[floor_point])
        else:
            xu, yu = ceil_point - floor_point, original_image[ceil_point] - original_image[floor_point]
            new_point = (((ratio * i) - floor_point) / xu) * yu + original_image[floor_point]
            new_array.append(new_point)
    print(f'Original Array: {original_image}')
    print(f'New Array: {new_array}')
