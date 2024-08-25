import pandas as pd
import matplotlib.pyplot as plt

# File path (make sure the path is correct)
filename = 'csv_outputs/scores20240825_174944.csv'

# Load the data into a DataFrame
df = pd.read_csv(filename, header=None)

# Separate predicted bids (columns 1 to 8) and true bids (columns 9 to 16)
predicted_bids = df.iloc[:, 1:9]
true_bids = df.iloc[:, 9:]

# Calculate the average predicted bids per round
average_predicted_bids = predicted_bids.mean(axis=0)

# Calculate the average true bids per round
average_true_bids = true_bids.mean(axis=0)

# Calculate the overall average of predicted bids
overall_average_predicted = predicted_bids.values.mean()

# Calculate the overall average of true bids
overall_average_true = true_bids.values.mean()

# Print the results
print("Average Predicted Bids per Round:")
print(average_predicted_bids)

print("\nAverage True Bids per Round:")
print(average_true_bids)

print("\nOverall Average Predicted Bids:")
print(overall_average_predicted)

print("\nOverall Average True Bids:")
print(overall_average_true)

# Plot the averages
plt.figure(figsize=(10, 6))

# Plot average predicted bids per round
plt.plot(range(2, 10), average_predicted_bids, marker='o', label='Average Predicted Bids', color='blue')

# Plot average true bids per round
plt.plot(range(2, 10), average_true_bids, marker='x', label='Average True Bids', color='orange')

# Plot overall average predicted bids
plt.axhline(y=overall_average_predicted, color='darkblue', linestyle='--', label='Overall Average Predicted Bids')

# Plot overall average true bids
plt.axhline(y=overall_average_true, color='darkorange', linestyle='--', label='Overall Average True Bids')

plt.xlabel('Round')
plt.ylabel('Average Bid')
plt.title('Average Predicted Bids vs Average True Bids per Round')
plt.xticks(range(2, 9))  # Set x-axis ticks to start from 2 to 8
plt.legend()
plt.grid(True)
plt.show()
