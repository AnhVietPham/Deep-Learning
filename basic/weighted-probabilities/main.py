import random
from random import choice, sample
import numpy as np


def weighted_choice(objects, weights):
    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random.random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]


if __name__ == "__main__":
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)
    print(np.multiply(x1, x2))
    print()
    print("=" * 50)
    print(choice("abcdefghij"))
    professions = ["scientist", "philosopher", "engineer", "priest"]
    print(professions)
    print(choice(("beginner", "intermediate", "advanced")))
    # rolling one die
    x = choice(range(1, 7))
    print("The dice shows: " + str(x))

    # rolling two dice:
    dice = sample(range(1, 7), 2)
    print("The two dice show: " + str(dice))

    print()
    print("=" * 50)
    weights = [0.2, 0.5, 0.3]
    cum_weights = [0] + list(np.cumsum(weights))
    print(cum_weights)

    print()
    print("=" * 50)
