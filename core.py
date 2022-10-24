import numpy as np
from math import pi
import scipy
from importlib import reload
import gradient
reload(gradient)

from gradient import reliable_gradient_field, point_grad
from utils import iround


EPS = 1e-3




def make_random(rand=None):
    if rand is None:
        rand = np.random.RandomState()
    elif np.isscalar(rand):
        rand = np.random.RandomState(rand)
    return rand


def ang_between(x, y):
    """counter clockwise rotation angle to convert x to y"""
    return np.arctan2(x[0], x[1]) - np.arctan2(y[0], y[1])


def rotation(x, theta):
    # counter clockwise rotation
    return np.array([
        x[0] * np.cos(theta) - x[1] * np.sin(theta),
        x[0] * np.sin(theta) + x[1] * np.cos(theta)
    ]
    )


def abscos(x):
    return np.abs(np.cos(x))


class RefractiveIndex:
    def __call__(self, wvlen):
        pass


class RefractiveIndexConst(RefractiveIndex):
    def __init__(self, const):
        self.const = const

    def __call__(self, wvlen):
        return self.const


def vec_normalize(x):
    return x / np.sqrt((x * x).sum())


def make_acute(ang_rad):
    ang = np.degrees(ang_rad)
    if (ang < -90) and (ang > -180):
        return -np.pi / 2, False
    elif (ang < -180) and (ang > -270):
        return np.pi / 2, False
    elif (ang > 90) and (ang <= 180):
        return np.pi / 2, False 
    elif (ang > 180) and (ang < 270):
        return -np.pi / 2, False
    else:
        return ang_rad, True
    

"""
lam = lam0 / n(lam0)
lam = nu / f
k = 2 * pi / lam = 2 * pi * n(lam0) / lam0 --> so everything can be expressed in terms of free space wavelength
"""

def single_ray_simulate(
        medium: np.ndarray,
        initial_direction: np.ndarray,
        initial_coord: np.ndarray,
        max_itr,
        n_void: RefractiveIndex,
        n_matter: RefractiveIndex,
        wvlen: float, # Note: This is the free space wavelength 
        pixel_length: float,
        grad_field=None,
        rand=None,
        propagate_norm: float = 1,
        power_tolerance: float = 1e-6
):  

    """
    Performs a single ray tracing simulation for a 2D binary medium.
    :param medium: 2D binary (0-1) medium. 1 represents material and 0 means void.
    :param initial_direction: 1 by 2 array of initial ray direction.
    :param initial_coord: 1 by 2 array of initial ray coordinate
    :param max_itr: maximum number of steps. Simulation ends if the ray energy fades, or it exits the medium 
    :param grad_field: the 2D pixel based surface gradient. If left as default None, 
        it will be calculated automatically from the medium configuration by calling gradient_field.
    :param rand: input random state object. If left as default None, will use np.random.RandomState()
    :param propagate_norm: the size of the propagation (step) vector.
    :return: a tuple containing (path, itr, coord, direction) where:
        path: 2D binary array that can be used to illustrate the path of ray
        itr: final iteration number
        coord: final ray coordinate
        direction: final ray direction vector
    """
    n_v = n_void(wvlen)
    n_m = n_matter(wvlen)
    
    # The dispersion is already taken into account by looking at free space wavelength and considering wave numbers
    wavenumber_v = 2 * np.pi * n_v / wvlen
    wavenumber_m = 2 * np.pi * n_m / wvlen
    power = 1

    if grad_field is None:
        # grad_field = gradient_field(medium, normalize=True)
        grad_field = reliable_gradient_field(medium, smooth_sigma=1, normalize=True)
    
    path = np.zeros_like(medium)
    num_rows, num_cols = medium.shape
    direction = np.array(initial_direction)
    coord = np.array(initial_coord)
    rand = make_random(rand)

    itr = 0
    next_coord = coord.copy() 


    def medium_of(c):
        try:
            return medium[iround(c[0]), iround(c[1])]
        except IndexError:
            return -1


    while True:
        itr += 1
        if (itr > max_itr) or (power < power_tolerance):
            break
        # normalizing direction to have a fixed setup size 
        direction = propagate_norm * direction / np.sqrt((direction * direction).sum()) 
        next_coord = next_coord + direction 
        if np.all(np.around(next_coord).astype(int) == np.around(coord).astype(int)): 
            continue
        if (
            (iround(next_coord[0]) >= num_rows) or 
            (iround(next_coord[0]) < 0) or 
            (iround(next_coord[1]) >= num_cols) or 
            (iround(next_coord[1]) <= 0)
        ): # falling outside the boundary. # TODO: at some point add boundary conditions for the sides.  
            break
        else:
            medium_now = medium[iround(coord[0]), iround(coord[1])]
            medium_next = medium[iround(next_coord[0]), iround(next_coord[1])]
            if medium_next == medium_now:
                wnum = wavenumber_v if (medium_now == 0) else wavenumber_m
                delta_move = np.sqrt(((next_coord - coord) ** 2).sum()) * pixel_length 
                power *= np.abs(np.exp(1j * wnum * delta_move)) ** 2  # exp(j k z) 
                coord = next_coord.copy()  
                path[iround(coord[0]), iround(coord[1])] = power
            else:
                n_now, n_next = (n_v, n_m) if medium_now == 0 else (n_m, n_v)

                grad = grad_field[:, iround(next_coord[0]), iround(next_coord[1])]
                gnorm = np.sqrt(np.sum(grad * grad)) # The norm of the grad vector

                if gnorm < EPS:
                    print("GRAD SINGULARITY")
                    grad = point_grad(medium, iround(coord[0]), iround(coord[1]))
                    if medium_now > 0:
                        grad = -grad
                    gnorm = np.sqrt(np.sum(grad * grad)) # The norm of the grad vector
                    

                pcoord = coord.copy()        
                coord = next_coord.copy()  
                path[iround(coord[0]), iround(coord[1])] = power

                # raise ValueError()
                    
                # n_now = n_m if medium_now == 1 else n_v # C++ equivalent: n_now = medium_now == 1 ? n_m : n_v
                # n_next = n_m if medium_next == 1 else n_v
                # grad = grad_field[:, round(next_coord[0]), round(next_coord[1])]  


                if gnorm > EPS:
                    surface_normal = grad / gnorm
                    surfce_normal_dot_dir = direction.dot(surface_normal) # inner product of direction and normal vectors
                    # ang_between of -direction and norm
                    theta_inc = np.arctan2(-direction[0], -direction[1]) - np.arctan2(surface_normal[0], surface_normal[1])
                    theta_inc_correct, is_acute = make_acute(theta_inc)
                    if not is_acute:
                        raise ValueError()
                        print(f"Warning: inaccurate gradient (obtuse angle). Manually correcting normal to {np.degrees(theta_inc_correct)}")    
                        theta_inc = theta_inc_correct
                        grad = rotation(-direction, theta_inc)
                        surface_normal = grad / np.sqrt((grad * grad).sum()) 
                    # Snell's law
                    theta_through = np.arcsin(np.abs(n_now / n_next) * np.sin(theta_inc))

                    if np.isnan(theta_through):  # Total Internal Reflection
                        direction = direction - 2 * (surface_normal.dot(direction)) * surface_normal  # Reflection formula
                        coord = next_coord + direction  
                        path[iround(coord[0]), iround(coord[1])] = power
                        next_coord = coord.copy()
                    else:
                        # Fresnel Equations
                        rs = (n_now * abscos(theta_inc) - n_next * abscos(theta_through)) / (n_now * abscos(theta_inc) + n_next * abscos(theta_through))
                        rp = (n_next * abscos(theta_inc) - n_now * abscos(theta_through)) / (n_next * abscos(theta_inc) + n_now * abscos(theta_through))
                        rho = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)
                        #assert rho < 1
                        assert rho - 1 < 1e-7
                        r = rand.uniform(0, 1)
                        if r < rho: # probabilistic reflection 
                            print("YAYYYYY")
                            direction = direction - 2 * (surface_normal.dot(direction)) * surface_normal  # Reflection formula
                            coord = next_coord + direction  
                            # raise ValueError()
                            # when direction is infenitesimal, continue until the ray pops out
                            num_count = 0
                            while True:
                                num_count += 1
                                coord = coord + direction
                                if medium_of(coord) == medium_now:
                                    break
                                if num_count > 3:
                                    raise AssertionError("Could not pop ray out of particle in a reasonable number of counts.")

                            path[iround(coord[0]), iround(coord[1])] = power
                            next_coord = coord.copy()
                        else:  # Refraction
                            n_direction = rotation(-surface_normal, -theta_through)
                            direction = n_direction
                            #direction = n1 / n2 * (direction + np.cos(theta_inc) * norm) - np.sqrt(
                            #    1 - (n1 / n2) ** 2 * (np.sin(theta_inc)) ** 2)  # Refraction function
                else:
                    print(f"ZERO GRADIENT {coord} {next_coord}")
                    raise ValueError(f"ZERO GRADIENT {coord} {next_coord}")


    return path, itr, coord, direction


def single_ray_opaque_simulate(
        medium: np.ndarray,
        initial_direction: np.ndarray,
        initial_coord: np.ndarray,
        max_itr,
        beta: float,
        grad_field=None,
        rand=None,
        propagate_norm: float = 1,
        power_tolerance: float = 1e-6
):  

    """
    Performs a single ray tracing simulation for a 2D binary medium with opaque particles (always reflect)
    with simple participating media behaviors expressed as a beta constant (beta = 4pi k / lambda)
    :param medium: 2D binary (0-1) medium. 1 represents material and 0 means void.
    :param initial_direction: 1 by 2 array of initial ray direction.
    :param initial_coord: 1 by 2 array of initial ray coordinate
    :param max_itr: maximum number of steps. Simulation ends if the ray energy fades, or it exits the medium 
    :param grad_field: the 2D pixel based surface gradient. If left as default None, 
        it will be calculated automatically from the medium configuration by calling gradient_field.
    :param rand: input random state object. If left as default None, will use np.random.RandomState()
    :param propagate_norm: the size of the propagation (step) vector.
    :return: a tuple containing (path, itr, coord, direction) where:
        path: 2D binary array that can be used to illustrate the path of ray
        itr: final iteration number
        coord: final ray coordinate
        direction: final ray direction vector
        step_sizes: array of peneteration depth values (step sizes) inside the medium
    """
    
    # The dispersion is already taken into account by looking at free space wavelength and considering wave numbers
    power = 1

    if grad_field is None:
        grad_field = reliable_gradient_field(medium, smooth_sigma=1, normalize=True)
    
    path = np.zeros_like(medium)
    num_rows, num_cols = medium.shape
    direction = np.array(initial_direction)
    coord = np.array(initial_coord)

    assert medium[coord[0], coord[1]] == 0, "Ray must start in medimum"
    rand = make_random(rand)

    itr = 0
    next_coord = coord.copy() 

    def medium_of(c):
        try:
            return medium[iround(c[0]), iround(c[1])]
        except IndexError:
            return -1
    

    strike_points = []

    while True:
        itr += 1
        if (itr > max_itr) or (power < power_tolerance):
            break
        # normalizing direction to have a fixed setup size 
        direction = propagate_norm * direction / np.sqrt((direction * direction).sum()) 
        next_coord = next_coord + direction 
        if np.all(np.around(next_coord).astype(int) == np.around(coord).astype(int)): 
            continue
        if (
            (iround(next_coord[0]) >= num_rows) or 
            (iround(next_coord[0]) < 0) or 
            (iround(next_coord[1]) >= num_cols) or 
            (iround(next_coord[1]) <= 0)
        ): # falling outside the boundary. # TODO: at some point add boundary conditions for the sides.  
            break
        else:
            medium_now = medium_of(coord)
            medium_next = medium_of(next_coord)
            if medium_next == medium_now:
                delta_move = np.sqrt(((next_coord - coord) ** 2).sum()) 
                power *= np.abs(np.exp(-beta * delta_move)) ** 2 
                coord = next_coord.copy()  
                path[iround(coord[0]), iround(coord[1])] = power
            else:
                strike_points.append(coord)
                grad = grad_field[:, iround(next_coord[0]), iround(next_coord[1])]
                gnorm = np.sqrt(np.sum(grad * grad)) # The norm of the grad vector
                if gnorm < EPS:
                    print("GRAD SINGULARITY")
                    grad = point_grad(medium, iround(coord[0]), iround(coord[1]))
                    if medium_now > 0:
                        grad = -grad
                    gnorm = np.sqrt(np.sum(grad * grad)) # The norm of the grad vector

                pcoord = coord.copy()        
                coord = next_coord.copy()  
                path[iround(coord[0]), iround(coord[1])] = power

                if gnorm > EPS:
                    surface_normal = grad / gnorm
                    surfce_normal_dot_dir = direction.dot(surface_normal) # inner product of direction and normal vectors
                    theta_inc = np.arctan2(-direction[0], -direction[1]) - np.arctan2(surface_normal[0], surface_normal[1])
                    theta_inc_correct, is_acute = make_acute(theta_inc)
                    if not is_acute:
                        raise ValueError()
                        print(f"Warning: inaccurate gradient (obtuse angle). Manually correcting normal to {np.degrees(theta_inc_correct)}")    
                        theta_inc = theta_inc_correct
                        grad = rotation(-direction, theta_inc)
                        surface_normal = grad / np.sqrt((grad * grad).sum()) 
                    # definite reflection
                    direction = direction - 2 * (surface_normal.dot(direction)) * surface_normal  # Reflection formula
                    # when direction is infenitesimal, continue until the ray pops out: 
                    num_cont = 0
                    while True:
                        num_cont += 1
                        coord = coord + direction  
                        if medium_of(coord) == medium_now:
                            break
                        if num_cont > 3:
                            raise AssertionError("Could not pop ray out of particle in a reasonable number of counts.")

                    path[iround(coord[0]), iround(coord[1])] = power
                    assert medium_of(coord) == medium_now, "Ray has ended up in material"
                    next_coord = coord.copy()
                else:
                    raise ValueError(f"ZERO GRADIENT {coord} {next_coord}")

        # f = lambda c: medium[iround(c[0]), iround(c[1])] == 1
        #assert medium[iround(coord[0]), iround(coord[1])] == 0, "Ray has ended up in material, something is wrong since matterial is opaque"
        # assert not(f(coord) and f(coord + direction)), "Ray has ended up in material, something is wrong since matterial is opaque"

    return path, itr, coord, direction, strike_points, power

