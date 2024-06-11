import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_file_size_with_trend(csv_file, parameter, parameter_value):
    data = []
    parameter_index = None
    other_parameter_index = None
    compressed_size_index = None

    # Read the CSV file and extract the relevant columns
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        # Get the indices of the chosen parameter and the compressed size columns
        parameter_index = header.index(parameter)
        compressed_size_index = header.index('Compressed Size')

        # Determine the other parameter (either 'Block Size' or 'Quality')
        other_parameter = 'Block Size' if parameter == 'Quality' else 'Quality'
        other_parameter_index = header.index(other_parameter)

        # Extract rows that match the parameter value
        for row in reader:
            if row[parameter_index] == str(parameter_value):
                data.append((float(row[other_parameter_index]), float(row[compressed_size_index])))

    if not data:
        print(f"No data found for {parameter} = {parameter_value}")
        return

    # Sort the data by the other parameter
    data.sort()

    # Split data into X and Y for plotting
    x_values, y_values = zip(*data)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', label='Compressed Size')

    # Add trend line for x_values >= 50
    x_values_array = np.array(x_values)
    y_values_array = np.array(y_values)
    mask = x_values_array >= 50
    x_trend = x_values_array[mask]
    y_trend = y_values_array[mask]

    if len(x_trend) > 1:  # Ensure there are enough points to fit a line
        coefficients = np.polyfit(x_trend, y_trend, 1)  # Linear fit
        trend_line = np.poly1d(coefficients)
        plt.plot(x_trend, trend_line(x_trend), color='red', linestyle='--', label='Trend Line (Quality ≥ 50)')

    # Add titles and labels
    plt.title(f'Compressed Size vs {other_parameter} for {parameter} = {parameter_value}')
    plt.xlabel(other_parameter)
    plt.ylabel('Compressed Size')
    plt.grid()
    plt.legend()

    # Show the plot
    plt.show()


# Example usage:
csv_file = 'fractal.csv'  # Replace with your CSV file path
parameter = 'Block Size'  # Replace with 'Quality' or 'Block Size'
parameter_value = 1  # Replace with the value you want to filter by

plot_file_size_with_trend(csv_file, parameter, parameter_value)
