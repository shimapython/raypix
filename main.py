import numpy as np


def normal_field(medium, normalize=False):
    """
    Calculates the normal field of a binary medium.

    :param medium: 2D binary array representing the medium
    :param normalize: boolean argument. If True, then normalizes the gradient to have norm 1 at every coordinate.
    :return: 2 * m * n array grad. grad[0, :, :] gives the 1st coordinate of the gradient at each pixel, and
    grad[1, :, :] gives the 2nd coordinate.

    It uses the following math:

    f(x + dx) - f(x) = g^T dx    

    dy = Dx g ==> g = (Dx^T Dx)^(-1) * Dx^T dy 

    Dx = 0  1
         0 -1
         1  0 
        -1  0
         1  1
         1 -1
        -1  1
        -1 -1
    """
    dy = np.array([
        np.roll(medium, shift=-1, axis=1),
        np.roll(medium, shift=1, axis=1),
        np.roll(medium, shift=-1, axis=0),
        np.roll(medium, shift=1, axis=0),
        np.roll(np.roll(medium, shift=-1, axis=0), shift=-1, axis=1),
        np.roll(np.roll(medium, shift=-1, axis=0), shift=1, axis=1),
        np.roll(np.roll(medium, shift=1, axis=0), shift=-1, axis=1),
        np.roll(np.roll(medium, shift=1, axis=0), shift=1, axis=1),
    ]
    )
    dy = np.array([medium - d for d in dy])
    dx = np.array([
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ])
    dx = np.linalg.inv(dx.T.dot(dx)).dot(dx.T)
    grad = np.array([
        np.array([a * b * medium for a, b in zip(dx[i, :], dy)]).sum(axis=0)
        for i in range(2)
    ])

    if normalize:
        norms = np.sqrt(grad ** 2).sum(axis=0)
        grad = np.array([np.nan_to_num(g / norms, 0) for g in grad])

    return grad


def ang_between(x, y):
    """counter clockwise rotation angle to convert x to y"""
    return np.arctan2(x[0], x[1]) - np.arctan2(y[0], y[1])


def rotation(x, theta):
    return np.array([
        x[0] * np.cos(theta) - x[1] * np.sin(theta),
        x[0] * np.sin(theta) + x[1] * np.cos(theta)
    ]
    )


def simulate(
        medium: np.ndarray,
        initial_direction: np.ndarray,
        initial_coord: np.ndarray,
        max_itr,
        normal=None,
        rand=None,
        reflection=0.5,
        propagate_norm: float = 1
):
    """
    Performs a single ray tracing simulation for a 2D binary medium.

    :param medium: 2D binary (0-1) medimum. 1 represents material and 0 means void.
    :param initial_direction: 1 by 2 array of initial ray direction.
    :param initial_coord: 1 by 2 array of initial ray coordinate
    :param max_itr: maximum number of steps. Simulation ends if the ray energy fades, or it exits the medium of
    #iterations >= max_itr
    :param normal: the 2D pixel based surface gradient. If left as default None, it will be calculated automatically
    from the medium configuration by calling normal_field.
    :param rand: input random state object. If left as default None, will use np.random.RandomState()
    :param reflection: the reflectivity constant.
    :param propagate_norm: the size of the propagation (step) vector.
    :return: a tuple containing (path, itr, coord, direction) where:
        path: 2D binary array that can be used to illustrate the path of ray
        itr: final iteration number
        coord: final ray coordinate
        direction: final ray direction vector
    """
    if normal is None:
        normal = normal_field(medium, normalize=True)
    path = 0 * medium
    # path = np.empty_like(medium)
    m, n = medium.shape
    direction = np.array(initial_direction)
    coord = np.array(initial_coord)
    if rand is None:
        rand = np.random.RandomState()
    path[int(coord[0]), int(coord[1])] = 1
    itr = 0
    next_coord = coord.copy()
    while True:
        itr += 1
        next_coord += direction
        if np.all(next_coord.astype(int) == coord.astype(int)):
            continue
        prev_medium = medium[int(coord[0]), int(coord[1])]
        direction = propagate_norm * direction / np.sqrt(np.sum(np.abs(direction)))
        if (int(next_coord[0]) >= m) or (int(next_coord[0]) < 0) or (int(next_coord[1]) >= n) or (
                int(next_coord[1]) <= 0):
            break
        else:
            coord = next_coord.copy()
            i, j = coord.astype(int)
            path[i, j] = 1
            if medium[i, j] != prev_medium:
                norm = normal[:, i, j]
                norm_norm = np.sqrt(np.sum(norm * norm))
                if norm_norm > 1e-3:
                    norm = norm / norm_norm
                    norm_dot_dir = direction.dot(norm)
                    # ang_between of -direction and norm
                    theta_inc = np.arctan2(-direction[0], -direction[1]) - np.arctan2(norm[0], norm[1])
                    theta_through = min(abs(theta_inc) / 2, np.pi / 4 - abs(theta_inc) / 2)
                    if theta_inc < 0:
                        theta_rot = np.pi / 2 - abs(theta_inc) - abs(theta_through)
                    else:
                        theta_rot = abs(theta_through) + abs(theta_inc) - np.pi / 2
                    if rand.uniform(0, 1) < reflection:
                        direction = direction - 2 * norm_dot_dir * norm
                        if np.sum(np.abs(direction)) < 1e-2:
                            direction = norm
                    # else:
                    #    # direction = rotation(direction, theta_rot)
                    #    direction = np.array([
                    #        direction[0] * np.cos(theta_rot) - direction[1] * np.sin(theta_rot),
                    #        direction[0] * np.sin(theta_rot) + direction[1] * np.cos(theta_rot)
                    #        ]
                    #    )
                else:
                    if medium[i, j]:
                        raise ValueError()

        if itr > max_itr:
            break

    return path, itr, coord, direction


def example_grad_plot():
    import pylab as plt
    medium = np.zeros([50, 100])
    medium[20:30, 20:30] = 1
    grad = normal_field(medium)
    x, y = plt.meshgrid(np.arange(medium.shape[1]), np.arange(medium.shape[0]))
    plt.imshow(medium)
    plt.quiver(x, y, grad[1], -grad[0], scale=10, width=0.002)
    plt.show()


def example_simulation():
    import pylab as plt
    import time
    med = (plt.imread("./medium.jpg")[:, :, 0] > 200).astype(int)
    normal = normal_field(med, normalize=True)
    rand = np.random.RandomState(seed=10)

    start_time = time.time()
    path, itr, _, _ = simulate(
        med,
        initial_direction=[1, 1],
        initial_coord=np.array([med.shape[0] / 2, med.shape[1] / 2]),
        max_itr=10000,
        normal=normal,
        rand=rand,
        reflection=1,
        propagate_norm=0.5
    )
    end_time = time.time()
    print(f"simulation finished in {end_time - start_time} seconds.\n#iterations:{itr}.")
    plt.imshow(med, cmap="gray")
    plt.imshow(path, alpha=0.8)
    plt.show()


if __name__ == "__main__":
    example_simulation()
