import os
import pandas as pd

# Start with an empty DataFrame
all_data = pd.DataFrame()

# Go through each subfolder in the current directory
for subfolder in os.listdir('../data/raw/dump'):
    # Ensure we're only looking at directories
    if os.path.isdir(subfolder):
        # Split subfolder name into components, accounting for underscores and spaces
        parts = subfolder.split(' ', 1)
        name_posture_trial, timestamp = parts[0], parts[1:]
        split_name = name_posture_trial.split('_')
        if (len(split_name) == 3):
            name, posture, trial = split_name
            if posture == "bedfall":
                posture = "bed_fall"
            if posture == "chairfall":
                posture = "chair_fall"
        else:
            posture = split_name[1] + split_name[2]
            name = split_name[0]
            trial = split_name[3]
        trial = int(trial)  # Convert trial number to int

        # Go through each CSV file in the subfolder
        for filename in os.listdir(subfolder):
            # Only process CSV files
            if filename.endswith('.csv'):
                # Construct the full path to the file
                filepath = os.path.join(subfolder, filename)

                # Load the file into a DataFrame
                data = pd.DataFrame(pd.read_csv(filepath))

                # Rename columns as 'measurement_submeasurement'
                data.columns = [f'{filename[:-4]}_{col.replace(" ", "_").lower()}' for col in data.columns]

                # Add the information from the subfolder name
                data['name'] = name
                data['posture'] = posture
                data['trial'] = trial
                data['timestamp'] = ' '.join(timestamp)  # Join timestamp components into a single string

                # Append this data to the full DataFrame

                all_data = pd.concat([all_data, data], ignore_index=True)

# Save the DataFrame to a CSV file
all_data.to_csv('../data/raw/all_data.csv', index=False)
