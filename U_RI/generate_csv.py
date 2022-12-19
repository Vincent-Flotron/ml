import csv
import matplotlib.pyplot as plt
import csv
import random
import math

# Set the parameters for the data
num_samples = 1000000  # The number of samples to generate
current_min = 0.001  # The minimum value for the random current values
current_max = 100  # The maximum value for the random current values
resistance_start = 0  # The starting resistance value
resistance_step = 100  # The step size for the resistance value

# Create an empty list to store the data
data = []

# Generate the data
resistance = resistance_start
for i in range(num_samples):
    # Generate a random current value
    current = random.uniform(current_min, current_max)
    # Calculate the tension
    tension = current * resistance
    # Add the data point to the list
    data.append([current, resistance, tension])
    # Increase the resistance value
    resistance = resistance_start + int(i/10000) * resistance_step

# Write the data to a CSV file
with open('data.csv', 'w', newline='') as csvfile:
    # Create the CSV writer
    writer = csv.writer(csvfile)
    # Write the column names as the first row
    writer.writerow(['Current', 'Resistance', 'Tension'])
    # Write the data rows
    writer.writerows(data)


# DISPLAY

# Prompt the user for the name of the csv file
# filename = input('Enter the name of the csv file: ')
filename = 'data.csv'

# Open the CSV file and read the data into a list
with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Extract the column names and the data rows from the list
column_names = data[0]
data_rows = data[1:]

# Get the number of columns in the data
num_columns = len(column_names)

# Extract the data from the list and convert it to lists of floats
column_data = []
for i in range(num_columns):
    column_data.append([float(row[i]) for row in data_rows])

# Create the figure and the subplots
fig, axs = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 6))

# Plot the data in each subplot
for i, ax in enumerate(axs):
    ax.plot(column_data[i], 'x')
    ax.set_title(column_names[i])

# Show the plot
plt.show()
