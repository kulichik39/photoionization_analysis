import numpy as np


# NOTE(anton):
# This is a repurposing of a quick and dirty script for
# generating a knot point sequence and recreating the old
# 'param.dat' input file for simulation box parameters.

def get_knotpoint_sequence_from_params(params_dict):
    debug_print_angle = False  # This can be used to print out the knot
    # sequence and make sure angle is constant in ECS region

    # Other parameters
    Z = params_dict["nuclear_charge_Z"]  # argon
    bspline_order_k = params_dict[
        "bspline_order_k"]  # This is for small component, large component
    # is automatically set to bspline_order_k-1 in generation of param.dat
    num_total_bsplines = params_dict["total_number_of_bsplines"]  # Or "physical" knotpoints

    # Grid start/end
    grid_start_point = params_dict["grid_start_point"]
    grid_end_point = params_dict["grid_end_point"]

    # First region
    first_non_zero_point = params_dict["first_non_zero_point"]
    second_non_zero_point = params_dict["second_point_multiplier"] * first_non_zero_point
    end_point_of_start_exp_region = params_dict["end_point_of_inner_region"]
    num_start_exponential_points = params_dict["inner_region_number_of_exponential_points"]

    # Mid region
    num_mid_linear_points = params_dict["mid_region_number_of_linear_points"]

    # End region (exterior complex scaling region (ECS))
    num_end_smooth_points = params_dict["ecs_region_number_of_points"]
    start_of_complex_scaling_point = params_dict["ecs_region_starting_point"]
    end_imaginary_coordinate = params_dict["ecs_end_imaginary_coordinate"]
    start_imaginary_coordinate = 0.0

    # We have bspline_order_k-1 ghost points at start and end
    num_total_knotpoints = num_total_bsplines + 2 * (bspline_order_k - 1)

    # print("num_start_exponential_points + num_mid_linear_points + num_end_smooth_points: ")
    pointsum = num_start_exponential_points + num_mid_linear_points + num_end_smooth_points
    print("%i + %i + %i = %i total points (also number of Bsplines)" % (
        num_start_exponential_points, num_mid_linear_points, num_end_smooth_points, pointsum))
    print("including %i ghost points (total length of knotpoint sequence): %i" % (
        2 * (bspline_order_k - 1), num_total_knotpoints))

    size_complex_region = grid_end_point - start_of_complex_scaling_point
    ang_rad = np.arctan(end_imaginary_coordinate / size_complex_region)
    ang_deg = np.rad2deg(ang_rad)
    print("Complex rotation angle arctan(%f/%f) = %f rad, %f deg" % (
        end_imaginary_coordinate, size_complex_region, ang_rad, ang_deg))

    knots = np.zeros(num_total_knotpoints, dtype=np.double)
    imag_knots = np.zeros(num_total_knotpoints, dtype=np.double)

    # set ghost points
    knots[0:bspline_order_k] = grid_start_point
    knots[(num_total_knotpoints - bspline_order_k):num_total_knotpoints] = grid_end_point
    imag_knots[(num_total_knotpoints - bspline_order_k):num_total_knotpoints] = end_imaginary_coordinate
    end_physical_point_index = (num_total_knotpoints - bspline_order_k) + 1

    # first non zero point
    first_non_zero_index = bspline_order_k
    knots[first_non_zero_index] = first_non_zero_point

    # second non zero point
    second_non_zero_index = first_non_zero_index + 1
    knots[second_non_zero_index] = second_non_zero_point

    # Start exponential region
    remaining_start_exp_points = num_start_exponential_points - 2
    start_exp_points_index = second_non_zero_index + 1
    end_exp_points_index = start_exp_points_index + remaining_start_exp_points
    # To make our life easy we start at the second nonzero point and distribute to the end point.
    knots[start_exp_points_index - 1:end_exp_points_index] = \
        np.logspace(np.log10(second_non_zero_point), np.log10(end_point_of_start_exp_region),
                    remaining_start_exp_points + 1, endpoint=True)

    start_exp_range = knots[start_exp_points_index - 1:end_exp_points_index]
    start_exp_range_plus1 = len(knots[start_exp_points_index - 1:end_exp_points_index]) + 1
    # print("start_exp_range_plus_2", start_exp_range_plus1)

    # Mid linear region
    linear_mid_region_start_index = end_exp_points_index - 2
    linear_mid_region_end_index = linear_mid_region_start_index + num_mid_linear_points
    knots[linear_mid_region_start_index:linear_mid_region_end_index] = \
        np.linspace(end_point_of_start_exp_region, start_of_complex_scaling_point, num_mid_linear_points)
    linear_mid_range = knots[linear_mid_region_start_index:linear_mid_region_end_index]
    # print("linear mid range:", len(linear_mid_range))

    # End region
    end_region_start_index = linear_mid_region_end_index - 1
    end_region_end_index = end_physical_point_index
    # print(knots[end_region_start_index], knots[end_region_end_index])

    # make sure two points after complex rotation has linear spacing
    linear_spacing = knots[end_region_start_index] - knots[end_region_start_index - 1]
    # imag_linear_spacing = end_imaginary_coordinate/(size_complex_region)

    knots[end_region_start_index + 1] = knots[end_region_start_index] + linear_spacing
    knots[end_region_start_index + 2] = knots[end_region_start_index] + 2 * linear_spacing

    imag_end_region_start_index = end_region_start_index
    # imag_knots[end_region_start_index + 1] = imag_knots[end_region_start_index] + imag_linear_spacing
    # imag_knots[end_region_start_index + 2] = imag_knots[end_region_start_index] + 2*imag_linear_spacing

    # shift index for logspace
    end_region_start_index += 2
    knots[end_region_start_index:end_region_end_index] = \
        np.logspace(np.log10(knots[end_region_start_index]), np.log10(grid_end_point), num_end_smooth_points)

    # print(knots[end_region_start_index-1], knots[end_region_start_index-2])
    b = (knots[end_region_start_index - 1] - start_of_complex_scaling_point)
    # print(b)
    start_point_for_log_imag = np.tan(ang_rad) * b
    # print(start_point_for_log_imag)
    imag_knots[end_region_start_index - 1] = start_point_for_log_imag
    for i in range(num_end_smooth_points):
        ii = i + end_region_start_index
        b = (knots[ii] - start_of_complex_scaling_point)
        imag_knots[ii] = np.tan(ang_rad) * (b)

    end_region_range = knots[end_region_start_index:end_region_end_index]
    end_region_range_len = len(end_region_range)

    if (debug_print_angle):
        for i in range(len(knots)):
            angle = 0.0
            j = 1
            start_coord_knots = knots[end_region_start_index - 2]
            if (i > end_region_start_index - 2 and i < end_region_end_index):
                a = imag_knots[i]
                b = knots[i] - start_coord_knots
                angle = np.arctan(a / b)
                # print(knots[i], start_coord_knots, b)
                # print(a, start_imaginary_coordinate)
                # print(angle)
                # print("\n")
                j += 1

            print(knots[i], "\t", imag_knots[i], "\t", angle)

    out_array = np.zeros((len(knots), 2))
    out_array[:, 0] = knots
    out_array[:, 1] = imag_knots
    return out_array, start_imaginary_coordinate
    # np.savetxt(knotsequence_filename, out_array, delimiter="    ", fmt='%1.13e')


def write_box_parameters_to_file(file1, params_dict, start_imaginary_coordinate):
    num_start_exponential_points = params_dict["inner_region_number_of_exponential_points"]
    grid_start_point = params_dict["grid_start_point"]
    grid_end_point = params_dict["grid_end_point"]
    bspline_order_k = params_dict[
        "bspline_order_k"]  # This is for small component, large component is automatically set to bspline_order_k-1 in generation of param.dat
    num_total_bsplines = params_dict["total_number_of_bsplines"]  # Or "physical" knotpoints
    num_mid_linear_points = params_dict["mid_region_number_of_linear_points"]

    # End region (exterior complex scaling region (ECS))
    num_end_smooth_points = params_dict["ecs_region_number_of_points"]
    start_of_complex_scaling_point = params_dict["ecs_region_starting_point"]
    end_imaginary_coordinate = params_dict["ecs_end_imaginary_coordinate"]

    file1.write("### Order of the first set of b-splines (for large component). ('x') \n")
    file1.write(str(bspline_order_k - 1) + "\n")
    file1.write("#### Order of the second set of b-splines (for small component). ('x') \n")
    file1.write(str(bspline_order_k) + "\n")
    file1.write("#### Number of points, start point to end point. ('x')\n")
    file1.write("%i \n" % (num_total_bsplines))
    file1.write("#### Convergence parameter Hartree-Fock. ('x.x')\n"
                "0.00000001\n"
                "#### Acceleration parameter Hartree-Fock ('x.x')\n"
                "0.20\n"
                "#### Inverse of fine structure constant. ('x.x')\n"
                "137.035999074")
