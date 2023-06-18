import pandas as pd

from Python3Code.Chapter3.DataTransformation import LowPassFilter
from sklearn.preprocessing import normalize


def lowpass_filter(df, columns):

    for col in columns:
        print("Lowpass for: " + col)
        # Lowpass filter sensory columns to remove walking pattern
        lowfilter = LowPassFilter()
        df = lowfilter.low_pass_filter(df, col, 100, 0.5)

    return df


def normalize_columns(df, cols):
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df


def ffill_columns(df, cols):
    for col in cols:
        df[col] = df[col].fillna(method='ffill')

    return df


def clean():
    df = pd.read_csv("data/raw/all_data.csv")

    columns_to_drop = [
        'Light_illuminance_(lx)',
        'Light_time_(s)',
        'Pressure_pressure_(hpa)',
        'Pressure_time_(s)',
        'Proximity_distance_(cm)',
        'Proximity_time_(s)',
        'timestamp'
    ]

    df = df.drop(columns=columns_to_drop)

    sensors = ["Accelerometer", "Gyroscope", "Light", "Linear Acceleration", "Magnetometer", "Pressure", "Proximity"]

    # set the location to the last recorded location
    df = ffill_columns(df, list(filter(lambda c: c.startswith("Location"), df.columns)))

    sensory_columns = [col for col in df.columns if col.startswith(tuple(sensors))]
    # normalize columns
    df = normalize_columns(df, sensory_columns)

    # make a lowpass filtering of the sensor columns
    df = lowpass_filter(df, sensory_columns)

    df.to_csv("data/cleaned/clean.csv")




if __name__ == '__main__':
    clean()
