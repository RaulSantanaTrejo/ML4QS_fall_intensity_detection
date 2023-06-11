import pandas as pd

from Python3Code.Chapter3.DataTransformation import LowPassFilter


def lowpass_filter(df, columns):

    for col in columns:
        print("Lowpass for: " + col)
        # Lowpass filter sensory columns to remove walking pattern
        lowfilter = LowPassFilter()
        df = lowfilter.low_pass_filter(df, col, 100, 0.5)

    return df


def clean():
    df = pd.read_csv("data/raw/all_data.csv")
    sensors = ["Accelerometer", "Gyroscope", "Light", "Linear Acceleration", "Magnetometer", "Pressure", "Proximity"]

    # make a lowpass filtering of the sensor columns
    sensory_columns = [col for col in df.columns if col.startswith(tuple(sensors))]
    df = lowpass_filter(df, sensory_columns)

    df.to_csv("data/cleaned/clean.csv")





if __name__ == '__main__':
    clean()