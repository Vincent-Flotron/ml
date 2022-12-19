import pandas as pd
from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
# from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the data from the CSV file
# df = pd.read_csv('my_csv.csv', sep=';')
df = pd.read_csv('data.csv', sep=',')


# Split the data into features and target
X = df[['Current', 'Resistance']]
y = df['Tension']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the model and train it on the training data
# model = LinearRegression()
# model = MLPClassifier(random_state=1, max_iter=300)
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = model.score(X_test, y_test)

print(f'Test score: {score:.2f}')

# Load the test data from the CSV file
# df_test = pd.read_csv('my_csv_test.csv', sep=';')
df_test = pd.read_csv('my_csv_test.csv', sep=';')

# Split the test data into features
X_test = df_test[['Current', 'Resistance']]

# Make predictions using the trained model
predictions = model.predict(X_test)

# predictions = predictions.astype(float)

# Add the predictions as a new column to the df_test dataframe
df_test['Tension predicted'] = predictions

# Add the difference
df_test['diff'] = df_test['Tension'] - df_test['Tension predicted']

# Save the df_test dataframe to a CSV file
df_test.to_csv('my_csv_test2.csv', index=False, sep=';')

# print(df_test)

# Extract the Tension and Tension predicted columns from the df_test dataframe
tension = df_test['Tension']
tension_predicted = df_test['Tension predicted']

# Plot
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

# Plot the Tension and Tension predicted data in the first subplot
ax1.plot(tension, label='Tension')
ax1.plot(tension_predicted, label='Tension predicted')
ax1.legend()

# Leave the second subplot empty
# Plot the Tension and Tension predicted data in the first subplot
ax2.plot(df_test['diff'], label='diff')
ax2.legend()

# Show the plot
plt.show()

# # Create the histogram
# data = df_test['Tension predicted']
# plt.hist(data, linewidth=0.9)

# # Add a title and x and y labels
# plt.title('Histogram of Tension predicted')
# plt.xlabel('Tension predicted')
# plt.ylabel('Frequency')

# # Show the plot
# plt.show()