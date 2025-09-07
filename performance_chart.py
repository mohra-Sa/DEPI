import pandas as pd
import matplotlib.pyplot as plt
import os # <-- We added this library

# ------------------ Part 1: Load Data ------------------
file_path = 'f:/progect data engineer/Students_clean (1).csv'
df = pd.read_csv(file_path)

# ------------------ Part 2: Prepare Data ------------------
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df['month'] = df['date'].dt.to_period('M')
performance_trends = df.groupby(['month', 'performance']).size().unstack(fill_value=0)

if 'Low' not in performance_trends.columns:
    performance_trends = performance_trends[['High', 'Medium']]

print("----------- Monthly Data Ready for Plotting -----------")
print(performance_trends)


# ------------------ Part 3: Create and Save the Plot ------------------

# Create the plot
performance_trends.plot(kind='bar', figsize=(12, 7), width=0.8)

# Add titles and labels
plt.title('Monthly Student Performance Trends', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Performance Level')
plt.tight_layout()

# --- New modification here ---
# Get the directory where the Python script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Join the script directory with the image name to create a full path
output_file_path = os.path.join(script_directory, 'performance_trends.png')

# Save the plot to the correct path
plt.savefig(output_file_path)

print(f"\nSuccess! 🎉 The chart has been saved to the correct path: {output_file_path}")