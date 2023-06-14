import os
import pandas as pd


if __name__ == '__main__':
    # Start with an empty DataFrame
    all_data = pd.DataFrame()

    # Go through each subfolder in the current directory
    for subfolder in os.listdir('../data/raw/dump'):
        subfolder = '../data/raw/dump/' + subfolder
        # Ensure we're only looking at directories
        if os.path.isdir(subfolder):
            # Split subfolder name into components, accounting for underscores and spaces
            parts = subfolder.split(' ', 1)
            name_posture_trial, timestamp = parts[0], parts[1:]
            split_name = name_posture_trial.split('_')

            if len(split_name) == 3:
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

            trial_data = pd.DataFrame()

            # Go through each CSV file in the subfolder
            for filename in os.listdir(subfolder):
                # Only process CSV files
                if filename.endswith('.csv'):

                    # Construct the full path to the file
                    filepath = os.path.join(subfolder, filename)

                    print(filepath)
                    # Load the file into a DataFrame
                    data = pd.DataFrame(pd.read_csv(filepath))

                    # Rename columns as 'measurement_submeasurement'
                    data.columns = [f'{filename[:-4]}_{col.replace(" ", "_").lower()}' for col in data.columns]

                    # Append this data to the full DataFrame
                    trial_data = pd.concat([trial_data, data], axis=1)

                    # Add the information from the subfolder name
            trial_data['name'] = name.split("/")[-1]
            trial_data['posture'] = posture
            trial_data['trial'] = trial
            trial_data['timestamp'] = ' '.join(timestamp)  # Join timestamp components into a single string

            all_data = pd.concat([all_data, trial_data], ignore_index=True, axis=0)

    # Save the DataFrame to a CSV file
    all_data.to_csv('../data/raw/all_data.csv', index=False)
