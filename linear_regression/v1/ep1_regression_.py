# Coronado González, Guillermo


import random as rnd
import math as math


# Multiplies two given matrices
def multiply_matrices(mat1, mat2):
    zip_mat2 = zip(*mat2)
    zip_mat2 = list(zip_mat2)

    return [[sum(float(element1) * float(element2) for element1, element2 in zip(row_1, col_2))
             for col_2 in zip_mat2] for row_1 in mat1]


# Returns the transpose of any given matrix
def get_matrix_transpose(mat):
    row_size = len(mat)
    column_size = len(mat[0])

    matrix_indices = {}

    for row_index in range(0, row_size):
        for column_index in range(0, column_size):
            value = mat[row_index][column_index]
            matrix_indices['{0}-{1}'.format(column_index, row_index)] = value

    mat_transpose = [None] * column_size

    for column_index in range(0, column_size):
        mat_transpose[column_index] = [None] * row_size
        for row_index in range(0, row_size):
            mat_transpose[column_index][row_index] = matrix_indices['{0}-{1}'.format(column_index, row_index)]

    return mat_transpose


# Converts data into 2D list
def data_1_into_2D_list():
    print("-" * 80)
    print("DATA TABLE SIZE: ")
    return make_2D_list(file_name, header_lines=0, footer_lines=0, strip=True, lower=True)


# Converts data into 2D list
def make_2D_list(filename, header_lines=0, footer_lines=0,
                 strip=True, lower=True):
    ''' Move data from file into a list of rows,
        where each row is a list of items of type string.
        Also eliminates empty rows.
    '''
    in_file = open(filename, 'r')
    # Input-output (xz) list of rows starts empty
    rows = []
    # For each line (string) in a data file
    for line in in_file:
        # Create row by splitting line into list of items (strings)
        row_list = line.split(",")
        if strip:
            # Eliminate prefix and suffix whitespace in each item
            row_list = [item.strip() for item in row_list]
        if lower:
            # Convert all letters into lowercase
            row_list = [item.lower() for item in row_list]
        # Add row_list into a list of lists
        rows.append(row_list)

    # Calculate number of lines stored in "rows" list
    R = len(rows)

    # Eliminate header lines and footer lines
    xz = rows[header_lines:R - footer_lines]
    print("Eliminated header rows = ", format(header_lines, '3d'))
    print("Eliminated footer rows = ", format(footer_lines, '3d'))

    # Calculate number of lines stored in "xz" list
    R = len(xz)

    # Eliminate empty rows from bottom to top
    eliminated = 0
    for Ri in range(R - 1, -1, -1):
        # If a row Ri s an empty string,
        if xz[Ri] == ['']:
            # delete it
            del xz[Ri]
            eliminated += 1
    print("Eliminated empty  rows = ", format(eliminated, '3d'),
          format(eliminated / R, '7.2%'), "\n")

    # Return 2D list of data items.
    return xz


# Gets data from file a writes into personal file
def extract_my_data():
    data_file = "airfoil_self_noise_.csv"
    in_file = open(data_file, 'r')
    my_file = open("airfoil_gcg_.csv", 'w')

    rnd.seed(69)
    for line in in_file:
        r = rnd.random()
        if r < 0.20:
            my_file.writelines(line.replace('\t', ','))

    in_file.close()
    my_file.close()


# Separates training data and test data
# Top 70% is training data
# Bottom 30$ is test data
def split_data(data):
    data_size = len(data)

    train_size = math.floor(data_size * 0.7)

    train_data = [None] * train_size
    test_data = [None] * (data_size - train_size)

    for data_index in range(0, data_size):
        if data_index < train_size:
            train_data[data_index] = data[data_index]
        else:
            test_data[data_index - train_size] = data[data_index]

    return train_data, test_data


# Splits original data and results vector (z vector)
def split_matrix_and_z_vector(train_data):
    data_size = len(train_data)
    matrix_a = [None] * data_size
    z_vector = [None] * data_size

    for train_index in range(0, data_size):
        matrix_a[train_index] = [1] + train_data[train_index][0:5]
        z_vector[train_index] = [train_data[train_index][5]]

    return matrix_a, z_vector


# Calls helper functions to train and test model
def split_train_and_test():
    data = data_1_into_2D_list()
    [train_data, test_data] = split_data(data)

    [matrix_a, z_vector] = split_matrix_and_z_vector(train_data)

    matrix_a_t = get_matrix_transpose(matrix_a)

    mat_prod = multiply_matrices(matrix_a_t, matrix_a)
    b_vector = multiply_matrices(matrix_a_t, z_vector)

    result_vector = solve_matrix_by_gauss(mat_prod, b_vector)

    accumulated_sum = 0

    for row in range(len(test_data)):
        sum_values = result_vector[0][0]
        for column in range(len(test_data[0]) - 1):
            sum_values += (result_vector[column + 1][0] * float(test_data[row][column]))
        dif = float(test_data[row][column + 1]) - sum_values
        accumulated_sum += (dif * dif)

    accumulated_sum /= len(test_data)

    smre = math.sqrt(accumulated_sum)

    return [result_vector, smre]


# Solves any given matrix and its output vector by Gauss method
def solve_matrix_by_gauss(inner_mat, output_vec):
    mat_size = len(inner_mat)
    mat_column_size = len(inner_mat[0])

    for pivot_index in range(0, mat_size):
        pivot = inner_mat[pivot_index][pivot_index]
        inverse_pivot = 1 / pivot
        output_vec[pivot_index][0] = output_vec[pivot_index][0] * inverse_pivot
        for index in range(0, mat_column_size):
            value = inner_mat[pivot_index][index]
            inner_mat[pivot_index][index] = value * inverse_pivot

        for row_index in range(pivot_index + 1, mat_size):
            neg_value = (-1) * (inner_mat[row_index][pivot_index])
            output_vec[row_index][0] += output_vec[pivot_index][0] * neg_value
            for column_index in range(pivot_index, mat_column_size):
                add_value = inner_mat[pivot_index][column_index] * neg_value
                inner_mat[row_index][column_index] += add_value

    for row in range(mat_size - 1, -1, -1):
        sum_values = 0
        for column in range(row + 1, mat_column_size):
            sum_values += (inner_mat[row][column] * output_vec[column][0])
        output_vec[row][0] -= sum_values

    return output_vec


if __name__ == "__main__":
    # extract_my_data()
    file_name = "airfoil_gcg_.csv"
    [result, error] = split_train_and_test()

    print('Result Vector: ')
    print(result)
    print('Result SMRE: ')
    print(error)
