import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tcn import compiled_tcn


def posture_only(group, col='Accelerometer_acceleration_y_(m/s^2)', seconds=1):
    # Find the maximum value of 'COL' and corresponding time value
    accelerometer_time = 'Accelerometer_time_(s)'
    max_value = group[col].max()
    max_time = group.loc[group[col] == max_value, accelerometer_time].values[0]

    # Define the time window
    window_start = max_time - seconds
    window_end = max_time + seconds

    # Retrieve rows within the time window
    window_rows = group.loc[
        (group[accelerometer_time] >= window_start) &
        (group[accelerometer_time] <= window_end)
        ]

    return window_rows


def prepare_data():
    # Step 1: Load "clean.csv" into a pandas dataframe
    df = pd.read_csv('../data/cleaned/non_lowpass.csv')

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[['posture']])

    filtered_df = df.filter(regex='^Magnetometer', axis=1)
    remaining_columns = df.drop(filtered_df.columns, axis=1)
    remaining_columns = remaining_columns.dropna()

    df = remaining_columns

    # Step 2: Group the rows by "name", "trial", and "posture"
    groups = df.groupby(['name', 'trial', 'posture'])

    # Step 3: Transform groups into a 2D array
    regressors_array = []
    labels_array = []
    for group_name, group_data in groups:
        group_data = posture_only(group_data)

        regressors = group_data.drop(columns=['name', 'trial', 'posture'])
        row_values = regressors.values.tolist()
        regressors_array.append(row_values)
        labels_array.append(encoder.transform(group_data[['posture']])[0])

    min_series_length = min([len(r) for r in regressors_array])
    print("Smallest size: ", min_series_length)
    for i, regressor in enumerate(regressors_array):
        regressors_array[i] = regressor[len(regressor)//2 - (min_series_length//2):len(regressor)//2 + (min_series_length//2)]

    return np.array(regressors_array), np.array(labels_array), encoder


def analyze_predictions(predictions, enc: OneHotEncoder, test_y):
    y_pred = enc.inverse_transform(predictions).flatten()
    y_true = enc.inverse_transform(test_y).flatten()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=enc.categories_)
    disp.plot()
    plt.show()


def run_task(inputs, outputs, enc):
    train_x, test_x, train_y, test_y = train_test_split(inputs, outputs, test_size=0.3)
    model = compiled_tcn(return_sequences=False,
                         num_feat=train_x.shape[2], #Changed, test_x should have 1 entry
                         nb_filters=24,
                         num_classes=8,
                         kernel_size=8,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         max_len=train_x.shape[1],
                         use_skip_connections=True,
                         output_len=test_y.shape[1]  # Changed from 27 to 8
                         )
    model.summary()
   # model.fit(train_x, train_y, epochs=2, validation_data = (test_x, test_y)) # removed batch
    model.fit(train_x, train_y.squeeze().argmax(axis=1), batch_size=45, epochs=1,
              validation_data=(test_x, test_y.squeeze().argmax(axis=1)))
    print(model.predict(test_x))
    y_raw_pred = model.predict(np.array(test_x))
    analyze_predictions(y_raw_pred, enc, test_y)


if __name__ == '__main__':
    inputs, outputs, encoder = prepare_data()
    run_task(inputs, outputs, encoder)
