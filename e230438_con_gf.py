import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split

# Function to combine 'DATE' and 'TIME' columns into a new datetime column
def combine_columns(row):
    return str(row['DATE']) + str(row['TIME']) + ':00'

# Main function for the E230438_CON_GF model
def e230438_con_GF():
    """
    @return: Power Generation predictions for given test data as an array

    """
    # Loop until a valid input is received for training option
    while True:
        will_train = input("Do you want to train a new model? [Y/N]: ")
        if will_train == "Y" or will_train == "N":
            break
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

    # If user chooses to train a new model
    if will_train == "Y":
        # Input the location of the training data file
        train_data = input("Please enter train data file's location:")

        # Read datetime and variable columns from Excel file
        df_datetime = pd.read_excel(train_data, usecols='A, B', dtype=str)
        df_variables = pd.read_excel(train_data, usecols='C:G')

        # Preprocess datetime columns
        df_datetime['TIME'] = df_datetime['TIME'].str.replace('24:00', '23:59')
        df_datetime['DATE'] = df_datetime['DATE'].str[:11]
        df_datetime['DATE'] = df_datetime.apply(combine_columns, axis=1)
        df_datetime.drop(['TIME'], axis=1, inplace=True)
        df_datetime['DATE'] = pd.to_datetime(df_datetime['DATE'], errors='coerce')

        # Combine datetime and variable columns
        df_dataset = pd.concat([df_datetime, df_variables], axis=1)
        df_dataset = df_dataset.set_index(['DATE'], drop=True)

        # Split the dataset into features (X) and target variable (Y)
        X = df_dataset.iloc[:, 0:-1].values
        Y = df_dataset.iloc[:, -1].values

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Build a neural network model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, input_dim=X_train.shape[1], activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train the model
        model.fit(X_train_scaled, Y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=2)

        # Save the trained model
        model.save("ann-model.h5")

        # Set the option to "R" to indicate that the model has been trained
        will_train = "R"

    # If user chooses not to train a new model or if the model has already been trained
    if will_train == "N" or will_train == "R":
        # Input the location of the test data file
        test_data = input("Please enter test data file's location:")

        # If the user chose not to train a new model, input the location of the existing model
        if will_train == "N":
            model_location = input("Please enter the model's location:")
        else:
            # Use the default model location if the user trained a new model
            model_location = "ann-model.h5"

        # Load the previously trained model
        prev_model = tf.keras.models.load_model(model_location)

        # Read the forecast data from the test file
        forecast_data = pd.read_excel(test_data, usecols='C:F')

        # Standardize the forecast data
        scaler = StandardScaler()
        new_data_scaled = scaler.fit_transform(forecast_data)

        # Make predictions using the trained model
        predictions = prev_model.predict(new_data_scaled)

        # Process predictions and store them in an array
        predictions_array = []
        for i, pred in enumerate(predictions):
            rounded_pred = abs(round(pred[0], 2))
            predictions_array.append(rounded_pred)

        # Return the array of predictions
        return predictions_array

# Entry point to the script
if __name__ == "__main__":
    e230438_con_GF()