import os
import sys 
import json

# set the maximum recurion limit
sys.setrecursionlimit(10000)
# the json path which contains the test cases for this problem
JSON_PATH = "test_cases.json"


def find_islands(grid, row, col):
    """This is recursive which check whether a perticular node is part of an island or not

    Args:
        grid (2D list): The grid to be iterated
        row (integer): current row number
        col (integer): current column number

    Returns:
        integer: 1 if the node is the part of an island otherwise 0
    """
    # the condition for a this source to a part of the island
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == "1":
        # when a land is visited make 0 
        grid[row][col] = "0"
        # iterate the right node
        find_islands(grid, row + 1, col)
        # iterate the botto, node
        find_islands(grid, row, col + 1)
        # iterate the left node
        find_islands(grid, row - 1, col)
        # iterate the top node
        find_islands(grid, row, col - 1)
        # if the condition is met it is part of a island
        return 1
    # if the condition doesn't met this is not a part of the island
    return 0


def calculate_number_of_islands(grid):
    """This function calculates the number of total island inside a given grid

    Args:
        grid (2D list): The grid to be iterated

    Returns:
        integer: Total number of island
    """    
    total_islands = 0
    #iterate the whole grid for possible islands
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            # increament number of islands the criteria
            total_islands += find_islands(grid, row, col)
    return total_islands
        


if __name__ == "__main__":
    # read the json file containing test cases
    with open(JSON_PATH, 'rb') as f:
        test_cases = json.load(f)
    # iterate through all test cases
    for key,value in test_cases.items():
        grid = []
        # converting the test case to a grid of 2D list
        for row in value:
            grid.append([i for i in row[0]])
        # calculate the output for each test case and print them        
        print(f"The output for {key} _is_ : {calculate_number_of_islands(grid)}")



    