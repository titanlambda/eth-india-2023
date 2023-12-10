import json
import math
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold

from tensorflow.keras.models import load_model
from joblib import dump, load

class RiskScoringAI:
    # Define a function to create the model, required for KerasRegressor
    def create_sequential_nn_model(self):
        # Define the model
        # model = Sequential()
        # model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(1))

        # Define the model with more layers
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))

        # Compile the model
        # model.compile(loss='mean_squared_error', optimizer='adam')

        # Compile the model with a custom learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def evaluate_nn_model_with_kfold(self, X, y):
        # Create the KerasRegressor with the model-creation function and the number of epochs
        keras_model = KerasRegressor(build_fn=self.create_sequential_nn_model, epochs=5, batch_size=32, verbose=0)
        # Define the k-fold cross validation
        kfold = KFold(n_splits=5, shuffle=True)
        # Perform the k-fold cross-validation
        results = cross_val_score(keras_model, X, y, cv=kfold)
        return results

    def fetch_data_from_db(self):
        # Connect to SQLite database
        conn = sqlite3.connect('risk_db.sqlite')
        query = "SELECT json_output FROM scan_findings"
        df = pd.read_sql_query(query, conn)
        return df

    def preprocess_data(self, df):
        impact_scores = {"Optimization": 1, "Informational": 2, "Low": 3, "Medium": 4, "High": 5}
        category_scores = {"contract_findings": 2, "node_modules_findings": 1}

        # Extract relevant information from JSON and convert to DataFrame
        data = pd.json_normalize(df['json_output'].apply(json.loads))
        # Initialize lists to store the extracted impacts and risk scores
        impacts = []
        finding_descriptions = []
        risk_scores = []
        # Extract 'Impact' from 'contract_findings' and 'node_modules_findings' and the corresponding 'risk_score'
        for i in range(len(data)):
            print(i)
            scan_result_contract_findings = data.loc[i, 'scan_result.contract_findings']
            scan_result_node_modules_findings = data.loc[i, 'scan_result.node_modules_findings']
            risk_score = data.loc[i, 'risk_score']

            inc = 0
            if not self.isNaN(scan_result_contract_findings):
                for finding in scan_result_contract_findings:
                    inc = inc + 1
                    print(inc)
                    impacts.append(finding['Impact'])
                    finding_descriptions.append(finding['Finding Description'])
                    impact_level = finding["Impact"]
                    temp_risk_score = (impact_scores[impact_level] * category_scores["contract_findings"])
                    risk_scores.append(temp_risk_score)

            if not self.isNaN(scan_result_node_modules_findings):
                for finding in scan_result_node_modules_findings:
                    impacts.append(finding['Impact'])
                    finding_descriptions.append(finding['Finding Description'])
                    impact_level = finding["Impact"]
                    temp_risk_score = (impact_scores[impact_level] * category_scores["node_modules_findings"])
                    risk_scores.append(temp_risk_score)

        # Convert the lists to a DataFrame
        df_impact = pd.DataFrame({'Impact': impacts, 'finding_descriptions': finding_descriptions, 'risk_score': risk_scores})
        return df_impact

    def encode_data(self, df_impact):
        # One-hot encode the 'Impact' feature
        encoder = OneHotEncoder(sparse=False)
        encoded_impact = encoder.fit_transform(df_impact[['Impact','finding_descriptions']])
        return encoded_impact

    def split_train_test_data(self, encoded_impact, df_impact):
        # Prepare the features (X) and target (y) for the model
        X = encoded_impact
        y = df_impact['risk_score'].values
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X, y, X_train, X_test, y_train, y_test

    def isNaN(self, data):
        return data != data

    def predict_risk_score(self, json_input, df_impact):
        # Load the models
        nn_model = load_model('sequential_model.h5')
        rf_model = load('random_forest_model.joblib')

        # Load the encoder
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(df_impact[['Impact','finding_descriptions']])  # df_impact is the DataFrame created earlier

        # Extract the 'Impact' from the JSON input
        impacts = []
        finding_descriptions = []
        for finding in json_input['scan_result']['contract_findings']:
            impacts.append(finding['Impact'])
            finding_descriptions.append(finding['Finding Description'])
        for finding in json_input['scan_result']['node_modules_findings']:
            impacts.append(finding['Impact'])
            finding_descriptions.append(finding['Finding Description'])

        # One-hot encode the 'Impact'
        encoded_impact = encoder.transform(np.array(impacts, finding_descriptions).reshape(-1, 1))

        # Use the models to make predictions
        nn_predictions = nn_model.predict(encoded_impact)
        rf_predictions = rf_model.predict(encoded_impact)
        print(f"nn_predictions: {nn_predictions}, rf_predictions: {rf_predictions}")

        # Calculate the average prediction
        average_prediction = (nn_predictions + rf_predictions) / 2

        return average_prediction

    def predict_with_random_forest_model(self, X_train, X_test, y_train, y_test):
        # Define the model
        model = RandomForestRegressor(n_estimators=30, random_state=42)
        # Train the model
        model.fit(X_train, y_train)
        # Use the model to make predictions on the test data
        predictions = model.predict(X_test)
        # Save the Random Forest model
        dump(model, 'random_forest_model.joblib')
        # Calculate the Mean Absolute Error (MAE)
        results_rf_mae = mean_absolute_error(y_test, predictions)
        # Calculate the Mean Squared Error (MSE)
        results_rf_mse = mean_squared_error(y_test, predictions)
        # Calculate the Root Mean Squared Error (RMSE)
        results_rf_rmse = math.sqrt(results_nn_mse)
        # Calculate the R^2 score
        results_rf_r2 = r2_score(y_test, predictions)
        return results_rf_mae, results_rf_mse, results_rf_rmse, results_rf_r2

    def predict_with_sequential_nn_model(self, X_train, X_test, y_train, y_test):
        nn_model = self.create_sequential_nn_model()
        # Train the model
        nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        # Use the model to make predictions on the test data
        nn_predictions = nn_model.predict(X_test)
        # Print the predictions
        print(nn_predictions)
        # Calculate the Mean Absolute Error (MAE)
        results_nn_mae = mean_absolute_error(y_test, nn_predictions)
        # Calculate the Mean Squared Error (MSE)
        results_nn_mse = mean_squared_error(y_test, nn_predictions)
        # Calculate the Root Mean Squared Error (RMSE)
        results_nn_rmse = math.sqrt(results_nn_mse)
        # Calculate the R^2 score
        results_nn_r2 = r2_score(y_test, nn_predictions)
        # Save the Sequential model
        nn_model.save('sequential_model.h5')
        return results_nn_mae, results_nn_mse, results_nn_rmse, results_nn_r2

    def evaluate_rf_model_with_kfold(self, X, y):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        # Define the k-fold cross validation
        kfold = KFold(n_splits=5, shuffle=True)
        # Perform the k-fold cross-validation
        results_rf_kfold = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        # Convert scores to positive (because cross_val_score calculates *negative* MSE)
        results_rf_kfold_mse_scores = -results_rf_kfold
        return results_rf_kfold_mse_scores


if __name__ == "__main__":
    risk_scoring_ai = RiskScoringAI()
    df = risk_scoring_ai.fetch_data_from_db()
    df_impact = risk_scoring_ai.preprocess_data(df)

    # json_input = {"blockchain_category": "POLY", "contract_address": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063", "network_id": "MAINNET", "risk_score": 40.0, "scan_result": {"contract_findings": [{"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44", "Finding Description": "Proxy.delegatedFwd(address,bytes) (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#18-44) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#20-43)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43", "Finding Description": "Proxy.delegatedFwd(address,bytes) (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#18-44) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#20-43)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "116, 117, 118, 119, 120, 121", "Finding Description": "UpgradableProxy.setProxyOwner(address) (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#116-121) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#118-120)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "118, 119, 120", "Finding Description": "UpgradableProxy.setProxyOwner(address) (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#116-121) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#118-120)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "139, 140, 141, 142, 143, 144", "Finding Description": "UpgradableProxy.setImplementation(address) (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#139-144) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#141-143)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "141, 142, 143", "Finding Description": "UpgradableProxy.setImplementation(address) (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#139-144) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#141-143)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156", "Finding Description": "UpgradableProxy.isContract(address) (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#146-156) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#152-154)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "152, 153, 154", "Finding Description": "UpgradableProxy.isContract(address) (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#146-156) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#152-154)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "101, 102, 103, 104, 105, 106, 107, 108", "Finding Description": "UpgradableProxy.loadImplementation() (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#101-108) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#104-106)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "104, 105, 106", "Finding Description": "UpgradableProxy.loadImplementation() (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#101-108) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#104-106)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "88, 89, 90, 91, 92, 93, 94, 95", "Finding Description": "UpgradableProxy.loadProxyOwner() (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#88-95) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#91-93)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "91, 92, 93", "Finding Description": "UpgradableProxy.loadProxyOwner() (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#88-95) uses assembly\n\t- INLINE ASM (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#91-93)\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "14", "Finding Description": "Pragma version0.6.6 (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#14) allows old versions\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "56", "Finding Description": "Pragma version0.6.6 (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#56) allows old versions\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "161", "Finding Description": "Pragma version0.6.6 (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#161) allows old versions\n"}, {"Impact": "Informational", "Source Filename": "crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol", "Line Numbers": "4", "Finding Description": "Pragma version0.6.6 (crytic-export/etherscan-contracts/0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063.polygonscan.com-UChildERC20Proxy.sol#4) allows old versions\n"}], "node_modules_findings": []}}
    # avg_pred_score = risk_scoring_ai.predict_risk_score(json_input, df_impact)
    # print(f"avg_pred_score: {avg_pred_score}")

    encoded_impact = risk_scoring_ai.encode_data(df_impact)
    X, y, X_train, X_test, y_train, y_test  = risk_scoring_ai.split_train_test_data(encoded_impact, df_impact)

    print("######## With SEQUENTIAL NN")
    results_nn_mae, results_nn_mse, results_nn_rmse, results_nn_r2 = risk_scoring_ai.predict_with_sequential_nn_model(X_train, X_test, y_train, y_test)

    # print("######## With NN KFOLD")
    # results_nn_kfold = risk_scoring_ai.evaluate_nn_model_with_kfold(X, y)

    # print("######## With RANDOM FOREST")
    # results_rf_mae, results_rf_mse, results_rf_rmse, results_rf_r2 = risk_scoring_ai.predict_with_random_forest_model(X_train, X_test, y_train, y_test)

    # print("######## With RANDOM FOREST KFOLD")
    # results_rf_kfold_mse_scores = risk_scoring_ai.evaluate_rf_model_with_kfold(X, y)


    print("#############################################")
    print("################ RESULTS ######################")
    print("#############################################")
    print(f"NN Mean Absolute Error (MAE): {results_nn_mae}")
    print(f"NN Mean Squared Error (MSE): {results_nn_mse}")
    print(f"NN Root Mean Squared Error (RMSE): {results_nn_rmse}")
    print(f"NN R^2 Score: {results_nn_r2}")
    # print(f"NN K-Fold Results: {results_nn_kfold.mean()} ({results_nn_kfold.std()}) MSE")
    # print(f"RF Mean Absolute Error (MAE): {results_rf_mae}")
    # print(f"RF Mean Squared Error (MSE): {results_rf_mse}")
    # print(f"RF Root Mean Squared Error (RMSE): {results_rf_rmse}")
    # print(f"RF R^2 Score: {results_rf_r2}")

    # Print the mean and standard deviation of the MSE from the cross-validation
    # print(f"Results: {results_rf_kfold_mse_scores.mean()} ({results_rf_kfold_mse_scores.std()}) MSE")