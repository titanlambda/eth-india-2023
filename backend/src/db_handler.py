import hashlib
import sqlite3
import zipfile
import os
import tempfile
import json
import io
from src.util import BlockchainCategory, PolygonNetworkType
import random


class DBHandler:
    def __init__(self):
        self.database = sqlite3.connect('risk_db.sqlite')
        self.database.execute('CREATE TABLE IF NOT EXISTS scan_findings (contract_adddress_hash TEXT PRIMARY KEY, json_output TEXT)')
        self.database.execute('CREATE TABLE IF NOT EXISTS contracts_ethereum (Txhash TEXT PRIMARY KEY, ContractAddress TEXT, ContractName TEXT, RiskScore REAL)')
        self.database.execute('CREATE TABLE IF NOT EXISTS contracts_polygon (Txhash TEXT PRIMARY KEY, ContractAddress TEXT, ContractName TEXT, RiskScore REAL)')

    def get_findings_from_db(self, contract_adddress_hash):
        cursor = self.database.execute('SELECT json_output FROM scan_findings WHERE contract_adddress_hash = ?',
                                       (contract_adddress_hash,))
        result = cursor.fetchone()
        if result is not None:
            return json.loads(result[0])
        else:
            return None

    def get_scan_results_poly_from_db(self):
        cursor = self.database.execute(
            'SELECT * FROM scan_findings WHERE json_output LIKE \'%"blockchain_category": "POLY",%\'')
        all_data = cursor.fetchall()
        result_data = []
        if all_data is not None:
            # for data_record in all_data:
            #     result_data.append(json.loads(data_record[0]))
            return all_data
        else:
            return None

    def get_scan_results_ethereum_from_db(self):
        cursor = self.database.execute(
            'SELECT * FROM scan_findings WHERE json_output LIKE \'%"blockchain_category": "ETHEREUM",%\'')
        all_data = cursor.fetchall()
        result_data = []
        if all_data is not None:
            # for data_record in all_data:
            #     result_data.append(json.loads(data_record[0]))
            return all_data
        else:
            return None

    def get_scan_results_poly_contract_hash_array_from_db(self):
        cursor = self.database.execute(
            'SELECT contract_adddress_hash FROM scan_findings WHERE json_output LIKE \'%"blockchain_category": "POLY",%\'')
        all_data = cursor.fetchall()
        result_data = []
        if all_data is not None:
            for data_record in all_data:
                result_data.append(data_record[0])
            return result_data
        else:
            return None


    def get_all_findings_from_db(self):
        cursor = self.database.execute('SELECT json_output FROM scan_findings')
        all_data = cursor.fetchall()
        result_data = []
        if all_data is not None:
            for data_record in all_data:
                result_data.append(json.loads(data_record[0]))
            return result_data
        else:
            return None

    def get_all_polygon_contract_summary_from_db(self):
        cursor = self.database.execute('''SELECT 
        	json_object(
                'ContractAddress', ContractAddress, 
                'ContractName', ContractName,
                'RiskScore', RiskScore
                )
            FROM contracts_polygon''')
        all_data = cursor.fetchall()

        # return all_data
        result_data = []
        if all_data is not None:
            for data_record in all_data:
                data_record = json.loads(data_record[0])
                result_data.append({
                        # "blockchain_category": BlockchainCategory.POLY.name,
                        # "network_id": PolygonNetworkType.MAINNET.name,
                        # "Txhash": data_record["Txhash"],
                        "ContractAddressme": data_record["ContractAddress"],
                        "ContractName": data_record["ContractName"],
                        "RiskScore": data_record["RiskScore"]
                })
            return result_data
        else:
            return None

    def get_all_ethereum_contract_summary_from_db(self):
        cursor = self.database.execute('''SELECT 
        	json_object(
                'ContractAddress', ContractAddress, 
                'ContractName', ContractName,
                'RiskScore', RiskScore
                )
            FROM contracts_ethereum''')
        all_data = cursor.fetchall()

        # return all_data
        result_data = []
        if all_data is not None:
            for data_record in all_data:
                data_record = json.loads(data_record[0])
                result_data.append({
                        # "blockchain_category": BlockchainCategory.ETHEREUM.name,
                        # "network_id": PolygonNetworkType.MAINNET.name,
                        # "Txhash": data_record["Txhash"],
                        "ContractAddressme": data_record["ContractAddress"],
                        "ContractName": data_record["ContractName"],
                        "RiskScore": data_record["RiskScore"]
                })
            return result_data
        else:
            return None

    def save_findings_to_db(self, contract_adddress_hash, json_output):
        json_dumps = json.dumps(json_output)
        self.database.execute('INSERT INTO scan_findings (contract_adddress_hash, json_output) VALUES (?, ?)', (contract_adddress_hash, json_dumps))
        self.database.commit()

    # Define a function to generate a random risk score
    def get_risk_score(self):
        return round(random.uniform(10, 65), 2)


    def update_ethereum_contracts_risk_score(self):
        # Fetch all records from the table
        cursor = self.database.execute("SELECT * FROM contracts_ethereum")
        rows = cursor.fetchall()

        # Loop through the records and update the RiskScore
        for row in rows:
            txhash = row[0]
            risk_score = self.get_risk_score()
            cursor.execute("UPDATE contracts_ethereum SET RiskScore = ? WHERE Txhash = ?", (risk_score, txhash))

        self.database.commit()


    def update_polygon_contracts_risk_score(self):
        # Fetch all records from the table
        cursor = self.database.execute("SELECT * FROM contracts_polygon")
        rows = cursor.fetchall()

        # Loop through the records and update the RiskScore
        for row in rows:
            txhash = row[0]
            risk_score = self.get_risk_score()
            cursor.execute("UPDATE contracts_polygon SET RiskScore = ? WHERE Txhash = ?", (risk_score, txhash))

        self.database.commit()


if __name__ == "__main__":
    print(DBHandler().get_scan_results_poly_contract_hash_array_from_db())
