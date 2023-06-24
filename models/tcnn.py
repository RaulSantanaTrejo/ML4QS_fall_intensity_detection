import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tcn import compiled_tcn, tcn_full_summary


def prepare_data():
    # Step 1: Load "clean.csv" into a pandas dataframe
    df = pd.read_csv('../data/cleaned/non_lowpass.csv')

    # Step 2: Group the rows by "name", "trial", and "posture"
    groups = df.groupby(['name', 'trial', 'posture'])

    # Step 3: Transform groups into a 2D array
    regressors_array = []
    labels_array = []
    for group_name, group_data in groups:
        regressors = group_data.drop(columns=['name', 'trial', 'posture'])
        row_values = regressors.values.tolist()
        regressors_array.append(row_values)
        labels_array = group_data['posture']

    return regressors_array, labels_array


def run_task(inputs, outputs):
    x_train, y_train, x_test, y_test = train_test_split(inputs, outputs, test_size=0.3)

    model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=10,
                         nb_filters=20,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         max_len=x_train[0:1].shape[1],
                         use_weight_norm=True,
                         use_skip_connections=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=100,
              validation_data=(x_test, y_test.squeeze().argmax(axis=1)))

    predictions = model.predict(x_test)
    accuracy = accuracy_score(predictions, y_test)
    print("Model accuracy: " + accuracy)


if __name__ == '__main__':
    inputs, outputs = prepare_data()
    run_task(inputs, outputs)
