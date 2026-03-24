import os
import pandas as pd
import csv
from pathlib import Path



data_path = Path(os.path.join("D:\\Faina", "Roots", "Sharon\\Hatzeva_all images","annotations\\all_additional_anns"))

# Find all CSV files with "TRL" in the filename
TRL_files = list(data_path.glob('*TRL*.csv'))

# Read and concatenate all matching CSV files
TRL_combined = pd.concat([pd.read_csv(file, header=None) for file in TRL_files],
                         ignore_index=True)

# Save to a new CSV file
output_path = data_path / 'combined_TRL.csv'
TRL_combined.to_csv(output_path, index=False, header=False)

# Find all CSV files with "pointsOutput" in the filename
points_files = list(data_path.glob('*pointsOutput*.csv'))

# Collect all rows from all files, no padding
all_rows = []

for file in points_files:
    with open(file, newline='') as f:
        reader = csv.reader(f)
        all_rows.extend(reader)

# Save to a new CSV file with jagged rows
output_file = data_path / 'combined_pointsOutput.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_rows)




