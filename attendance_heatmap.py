import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # We will use the seaborn library for the heatmap
import os

# ------------------ Part 1: Load Data ------------------
file_path = 'f:/progect data engineer/Students_clean (1).csv'
df = pd.read_csv(file_path)

# Convert date column and extract the month
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df['month'] = df['date'].dt.to_period('M')


# ------------------ Part 2: Prepare Data for the Heatmap ------------------
print("----------- Preparing Data for Heatmap -----------")

# We will pivot the data to have subjects as rows, months as columns,
# and the average attendance as the values.
attendance_pivot = df.groupby(['Subject', 'month'])['Attendance (%)'].mean().unstack()

print("Data ready for plotting:")
print(attendance_pivot)


# ------------------ Part 3: Create and Save the Heatmap ------------------
# Create a larger figure to make the heatmap easier to read
plt.figure(figsize=(15, 8))

# Create the heatmap using seaborn
# annot=True writes the attendance percentage inside each cell
# cmap sets the color scheme
sns.heatmap(attendance_pivot, annot=True, fmt=".1f", cmap='viridis')

# Add titles and labels
plt.title('Average Monthly Attendance (%) per Subject', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Subject', fontsize=12)
plt.tight_layout() # Adjusts the plot to make sure everything fits

# --- Save the heatmap to a file in the correct directory ---
script_directory = os.path.dirname(os.path.abspath(__file__))
output_file_name = 'attendance_heatmap.png'
output_file_path = os.path.join(script_directory, output_file_name)

plt.savefig(output_file_path)

print(f"\nSuccess! 🎉 Heatmap has been saved as '{output_file_name}' in your project folder.")