import numpy as np

if __name__ == "__main__":
    # indexing 1 dimension
    a1 = np.array([1, 2, 3, 4])
    print(a1)
    print(a1[::-1])

    # indexing 2 dimension
    a2 = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [7, 8, 9, 10]])
    print(a2)
    # Pick a row
    print(a2[0])
    print(a2[1])
    # Pick a column
    print(a2[:, 1])
    print(a2[::-1, 2])

    # Indexing 3 dimension
    a3 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                   [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
                   ])
    print(a3)
    # Pick one channel
    print(a3[0])
    # Pick row 2 of channel 2
    print(a3[2, 0])
    print(a3[:, :, ::-1])
