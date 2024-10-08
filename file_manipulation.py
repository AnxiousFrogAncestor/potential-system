import csv

def csv_open_to_matrix(filename):
    """Opens CSV file WITHOUT specifying file format, returns a 2d float array."""
    out_rows = []
    with open(f'{filename}.csv', newline='') as csvfile:
        file_input = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(file_input):
            float_ls = []
            for el in row:
                for char in el:
                    try:
                        num = float(char)
                        print(num)
                        float_ls.append(num)
                    except:
                        pass
                    finally:
                        print(f"done parsing {char}")
            print(float_ls)
            out_rows.append(float_ls)
    print("output rows", out_rows)
    return out_rows

def csv_write_to_matrix(matrix, filename):
    """Writes CSV file WITHOUT specifying file format, returns a 2d float array."""
    with open(f'{filename}.csv', "w", newline='') as csvfile:
        file_output = csv.writer(csvfile, delimiter=',', quotechar='|')
        for row in matrix:
            print(f"Writing {row} row into file")
            file_output.writerow(row)

def main():
    in_array = csv_open_to_matrix("example_file")
    sum(in_array[0])
    in_array[1] = [1.0, 2.0, 3.0]
    csv_write_to_matrix(in_array, "output_file")

#main()

