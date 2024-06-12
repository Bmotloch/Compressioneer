import csv
import matplotlib.pyplot as plt

# File path to your CSV file
csv_file_path = 'fractal.csv'

# Block sizes to be plotted
block_sizes_to_plot = [1, 8, 16, 32, 64]

# Data storage
data = {
    'Quality': [],
    'Block Size': [],
    'Compressed Size': []
}

# Read the CSV file
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data['Quality'].append(int(row['Quality']))
        data['Block Size'].append(int(row['Block Size']))
        data['Compressed Size'].append(float(row['Compressed Size']))


# Function to plot data for specific block sizes
def plot_data(data, block_sizes_to_plot):
    plt.figure(figsize=(11.69, 8.27))
    for block_size in block_sizes_to_plot:
        x_values = [data['Quality'][i] for i in range(len(data['Quality'])) if data['Block Size'][i] == block_size]
        y_values = [data['Compressed Size'][i] for i in range(len(data['Compressed Size'])) if
                    data['Block Size'][i] == block_size]

        plt.plot(x_values, y_values, linestyle='--', label=f'Block Size {block_size}')

    plt.xlabel('Quality')
    plt.ylabel('Compressed Size [KB]')
    plt.title('Compressed Size vs Quality for Different Block Sizes (fractal_tree.isa)')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()  # Adjust the layout to minimize empty space
    plt.show()


# Call the function to plot data
plot_data(data, block_sizes_to_plot)
