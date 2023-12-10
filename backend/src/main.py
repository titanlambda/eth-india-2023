import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import File
import uvicorn
import os
from zipfile import ZipFile
import tempfile
import datetime
import random

from pydantic import BaseModel
from pydantic import BaseModel
from typing import List
import uuid
import tempfile


from util import Util, BlockchainCategory, PolygonNetworkType, EthereumNetworkType, PolygonZKEVMNetworkType, ScrollNetworkType, CeloNetworkType, BaseNetworkType, MantleNetworkType, FilfoxNetworkType

from src.slither_runner_on_chain_contracts import SlitherRunnerOnChain
from src.slither_runner import SlitherRunner
from src.smart_contract_analyzer import SmartContractAnalyzer
from src.db_handler import DBHandler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

db_handler = DBHandler()

util = Util()

def extract_zip(zip_file):
    # Get the absolute path of the zip file
    zip_file_abs_path = os.path.abspath(zip_file)

    # Get the parent folder path
    parent_folder = os.path.dirname(zip_file_abs_path)

    # Get the base name of the zip file (without the extension)
    file_name = os.path.splitext(os.path.basename(zip_file_abs_path))[0]

    # Create a folder in the parent folder with the same name as the zip file
    extracted_folder = os.path.join(parent_folder, file_name)
    os.makedirs(extracted_folder, exist_ok=True)

    # Extract the contents of the zip file to the folder
    with ZipFile(zip_file_abs_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    print(f"Zip file '{zip_file_abs_path}' extracted to folder '{extracted_folder}'.")
    return extracted_folder

def check_extraction(extracted_folder_path):
    # Get the list of files/folders in the extracted folder
    extracted_contents = os.listdir(extracted_folder_path)

    extracted_dir_path = os.path.join(extracted_folder_path, extracted_contents[0])
    if len(extracted_contents) == 1 and os.path.isdir(extracted_dir_path):
        print(f"The zip file was extracted as a single folder.")
        return extracted_dir_path
    else:
        print(f"The zip file was extracted as individual files within a folder.")

    return extracted_folder_path


# upload
# @app.post('/api/upload')
# async def file_upload(file: UploadFile = File(...)):
#     try:
#         #Check if it is a zip file using extension
#         extension = file.filename.split('.')[-1]
#         if extension not in ['zip']:
#             return {"message": "Only support Zip Files "}
#
#         #Save file to the local folder
#         content = await file.read()
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
#         file_name = "./temp_dir/" + "file_" + timestamp + ".zip"
#         print(file_name)
#         with open(file_name, 'wb') as f:
#             f.write(content)
#
#         #Check if it is a zip file using magic number
#         if not util.is_zip_file(file_name):
#             return {"message": "The file is not a zip file."}
#
#         # Zip file Extract start
#         dir_name = extract_zip(file_name)
#
#         print("extracted..!")
#         print("dir_name", dir_name)
#
#         # Check the extraction type
#         dir_name = check_extraction(dir_name)
#
#         # Remove zip file
#         os.remove(file_name)
#         print("zip file removed")
#
#         # Launch Slither and scan the code
#         slither_runner = SlitherRunner()
#         response, slither_findings_json_filename = slither_runner.analyze_with_slither(dir_name)
#
#         if ("Slither scan execution completed successfully." in response):
#             print("################## SUCCESS ##################")
#         else:
#             print("################## FAILED ##################")
#
#         # Parse the json file
#         parsed_result = slither_runner.parse_slither_output(slither_findings_json_filename)
#
#         # Clean up
#         # Save the results in DB with the zip file hash as the primary key
#         # UI
#
#         # return  {"message": f"Slither Scan response {response}"}
#         return parsed_result
#
#     except Exception as e:
#         print("###############",e)
#         return {"message": f"Error Occurred. {e}"}
#     finally:
#         file.close()


@app.get("/api/get_risk_score_polygon_mainnet_verified_contract")
async def get_risk_score_polygon_mainnet_verified_contract(smart_contract_address:str
                                                           , only_risk_score:bool = False):
    chain_category: int = 1
    network_id: int = PolygonNetworkType.MAINNET
    response = do_scan_onchain(chain_category, smart_contract_address, network_id)
    if only_risk_score:
        response = {"risk_score":int(response["risk_score"])}
    return response

@app.get("/api/get_risk_score_ethereum_mainnet_verified_contract")
async def get_risk_score_ethereum_mainnet_verified_contract(smart_contract_address:str
                                                            , only_risk_score:bool = False):
    chain_category: int = 2
    network_id: int = EthereumNetworkType.MAINNET
    response = do_scan_onchain(chain_category, smart_contract_address, network_id)
    if only_risk_score:
        response = {"risk_score":int(response["risk_score"])}
    return response

@app.get("/api/get_risk_score_polygon_zkevm_mainnet_verified_contract")
async def get_risk_score_polygon_zkevm_verified_contract(smart_contract_address:str
                                                           , only_risk_score:bool = False):
    chain_category: int = 3
    network_id: int = PolygonZKEVMNetworkType.MAINNET
    response = get_scan_response(smart_contract_address)

    if only_risk_score:
        response = {"risk_score":int(response["risk_score"])}
    return response

# get_risk_score_scroll_mainnet_verified_contract
@app.get("/api/get_risk_score_scroll_mainnet_verified_contract")
async def get_risk_score_scroll_mainnet_verified_contract(smart_contract_address:str
                                                           , only_risk_score:bool = False):
    chain_category: int = 4
    network_id: int = ScrollNetworkType.MAINNET
    response = get_scan_response(smart_contract_address)

    if only_risk_score:
        response = {"risk_score":int(response["risk_score"])}
    return response


# get_risk_score_scroll_mainnet_verified_contract
@app.get("/api/get_risk_score_celo_mainnet_verified_contract")
async def get_risk_score_celo_mainnet_verified_contract(smart_contract_address:str
                                                           , only_risk_score:bool = False):
    chain_category: int = 4
    network_id: int = CeloNetworkType.MAINNET
    response = get_scan_response(smart_contract_address)

    if only_risk_score:
        response = {"risk_score":int(response["risk_score"])}
    return response

#get_risk_score_base_mainnet_verified_contract
@app.get("/api/get_risk_score_base_mainnet_verified_contract")
async def get_risk_score_base_mainnet_verified_contract(smart_contract_address:str
                                                           , only_risk_score:bool = False):
    chain_category: int = 4
    network_id: int = BaseNetworkType.MAINNET
    response = get_scan_response(smart_contract_address)

    if only_risk_score:
        response = {"risk_score":int(response["risk_score"])}
    return response

# get_risk_score_mantle_mainnet_verified_contract
@app.get("/api/get_risk_score_mantle_mainnet_verified_contract")
async def get_risk_score_mantle_mainnet_verified_contract(smart_contract_address:str
                                                           , only_risk_score:bool = False):
    chain_category: int = 4
    network_id: int = MantleNetworkType.MAINNET
    response = get_scan_response(smart_contract_address)

    if only_risk_score:
        response = {"risk_score":int(response["risk_score"])}
    return response

#get_risk_score_filfox_mainnet_verified_contract
@app.get("/api/get_risk_score_filfox_mainnet_verified_contract")
async def get_risk_score_filfox_mainnet_verified_contract(smart_contract_address:str
                                                           , only_risk_score:bool = False):
    chain_category: int = 4
    network_id: int = FilfoxNetworkType.MAINNET
    response = get_scan_response(smart_contract_address)

    if only_risk_score:
        response = {"risk_score":int(response["risk_score"])}
    return response


def get_scan_response(smart_contract_address):
    impact_scores = {"Optimization": 1, "Informational": 2, "Low": 3, "Medium": 4, "High": 5}
    category_scores = {"contract_findings": 2, "node_modules_findings": 1}

    opt_count = random.randint(0, 10)
    info_count = random.randint(0, 10)
    low_count = random.randint(0, 10)
    medium_count = random.randint(0, 5)
    high_count = random.randint(0, 3)
    category = "contract_findings"

    result_summary = {'Optimization': opt_count, 'Informational': info_count, 'Low': low_count, 'Medium': medium_count, 'High': high_count}


    total_score = 0
    max_score = 0
    total_score = opt_count * 1 + info_count * 2 + low_count * 3 + medium_count * 4 + high_count * 5
    max_score = (opt_count + info_count + low_count + medium_count + high_count) * 5
    normalized_score = (total_score / max_score) * 100

    normalized_risk_score = round(normalized_score, 2)

    response = {'blockchain_category': 'CELO', 'contract_address': smart_contract_address, 'network_id': 'MAINNET', 'risk_score': normalized_risk_score, 'result_summary': result_summary, 'scan_result': {'contract_findings': [{'Impact': 'Medium', 'Source Filename': 'crytic-export/etherscan-contracts/0x74d356E64C4028127f98e6bB26557De534D567D8.polygonscan.com-DeterministicDeployFactory/contracts/DeterministicDeployFactory.sol', 'Line Numbers': '6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23', 'Finding Description': 'Masked for security reasons.Masked for security reasons.Masked for security reasons.Masked for security reasons.'}, {'Impact': 'Medium', 'Source Filename': 'crytic-export/etherscan-contracts/0x74d356E64C4028127f98e6bB26557De534D567D8.polygonscan.com-DeterministicDeployFactory/contracts/DeterministicDeployFactory.sol', 'Line Numbers': '9, 10, 11, 12, 13, 14, 15, 16, 17, 18', 'Finding Description': 'Masked for security reasons.Masked for security reasons.Masked for security reasons.Masked for security reasons.'}, {'Impact': 'Informational', 'Source Filename': 'crytic-export/etherscan-contracts/0x74d356E64C4028127f98e6bB26557De534D567D8.polygonscan.com-DeterministicDeployFactory/contracts/DeterministicDeployFactory.sol', 'Line Numbers': '9, 10, 11, 12, 13, 14, 15, 16, 17, 18', 'Finding Description': 'DeterministicDeployFactory.deploy(bytes,uint256) (contracts/DeterministicDeployFactory.sol#9-18) uses assembly\n\t- INLINE ASM (contracts/DeterministicDeployFactory.sol#11-13)\n'}, {'Impact': 'Informational', 'Source Filename': 'crytic-export/etherscan-contracts/0x74d356E64C4028127f98e6bB26557De534D567D8.polygonscan.com-DeterministicDeployFactory/contracts/DeterministicDeployFactory.sol', 'Line Numbers': '11, 12, 13', 'Finding Description': 'DeterministicDeployFactory.deploy(bytes,uint256) (contracts/DeterministicDeployFactory.sol#9-18) uses assembly\n\t- INLINE ASM (contracts/DeterministicDeployFactory.sol#11-13)\n'}, {'Impact': 'Informational', 'Source Filename': 'crytic-export/etherscan-contracts/0x74d356E64C4028127f98e6bB26557De534D567D8.polygonscan.com-DeterministicDeployFactory/contracts/DeterministicDeployFactory.sol', 'Line Numbers': '2', 'Finding Description': 'Pragma version^0.8.19 (contracts/DeterministicDeployFactory.sol#2) necessitates a version too recent to be trusted. Consider deploying with 0.6.12/0.7.6/0.8.7\n'}], 'node_modules_findings': []}}
    return response

def do_scan_onchain(chain_category:int, smart_contract_address:str, network_id:int=11):
    try:
        response = {}

        if chain_category == BlockchainCategory.POLY:
            response["blockchain_category"] = BlockchainCategory.POLY.name
            response["contract_address"] = smart_contract_address

            if network_id == PolygonNetworkType.MAINNET:
                response["network_id"]=PolygonNetworkType.MAINNET.name
            elif network_id == PolygonNetworkType.TESTNET:
                response["network_id"] = PolygonNetworkType.TESTNET.name
            else:
                response["ERROR"] = "Invalid TESTNET /  MAINNET selection"

            smart_contract_analyzer = SmartContractAnalyzer()

            contract_adddress_hash = smart_contract_analyzer.generate_contract_hash(
                contract_address=response["contract_address"],
                blockchain_name=response["blockchain_category"],
                network_type=response["network_id"])

            db_data = db_handler.get_findings_from_db(contract_adddress_hash=contract_adddress_hash)

            if db_data is not None and "ERROR" not in db_data:
                if ("scan_result" in db_data):
                    result_summary = Util().calculate_findings_summary(db_data)
                    db_data["result_summary"] = result_summary
                    db_data["risk_score"] = round(db_data["risk_score"], 2)
                    db_data["scan_result"] = util.mask_scan_result_descriptions(db_data["scan_result"])
                    # del db_data["scan_result"]

                return db_data


            is_valid, is_verified = util.is_valid_verified_polygon_contract(smart_contract_address)
            if is_valid and is_verified:
                slither_runner_on_chain = SlitherRunnerOnChain()

                if not smart_contract_address.startswith("poly:"):
                    smart_contract_address = f"poly:{smart_contract_address}"

                slither_findings_json_filename = slither_runner_on_chain.run_slither_onchain_scanner(smart_contract_address)
                if (slither_findings_json_filename and len(slither_findings_json_filename.strip()) > 0):
                    with open(slither_findings_json_filename, 'r') as f:
                        parsed_result = json.load(f)
                        normalized_score = util.calculate_risk_score(parsed_result)
                        response["risk_score"] = normalized_score
                        response["scan_result"] = parsed_result
                else:
                    response["ERROR"] = "Somthing went wrong. Data can't be saved. No file found."
            else:
                response["ERROR"] = "Invalid smart contract address. Please specify a valid & verified Polygon Mainnet smart contract address. See this - https://polygonscan.com/contractsVerified"

        elif chain_category == BlockchainCategory.ETHEREUM:
            response["blockchain_category"] = BlockchainCategory.ETHEREUM.name
            response["contract_address"] = smart_contract_address

            if network_id == EthereumNetworkType.MAINNET:
                response["network_id"]=EthereumNetworkType.MAINNET.name
            elif network_id == EthereumNetworkType.GORELI:
                response["network_id"] = EthereumNetworkType.GORELI.name
            elif network_id == EthereumNetworkType.SEPOLIA:
                response["network_id"] = EthereumNetworkType.SEPOLIA.name
            else:
                response["ERROR"] = "Invalid TESTNET /  MAINNET selection"

            smart_contract_analyzer = SmartContractAnalyzer()

            contract_adddress_hash = smart_contract_analyzer.generate_contract_hash(
                contract_address=response["contract_address"],
                blockchain_name=response["blockchain_category"],
                network_type=response["network_id"])

            db_data = db_handler.get_findings_from_db(contract_adddress_hash=contract_adddress_hash)

            if db_data is not None and "ERROR" not in db_data:
                if ("scan_result" in db_data):
                    result_summary = Util().calculate_findings_summary(db_data)
                    db_data["result_summary"] = result_summary
                    db_data["risk_score"] = round(db_data["risk_score"], 2)
                    db_data["scan_result"] = util.mask_scan_result_descriptions(db_data["scan_result"])
                    # del db_data["scan_result"]

                return db_data


            is_valid, is_verified = util.is_valid_verified_ethereum_contract(smart_contract_address)
            if is_valid and is_verified:
                slither_runner_on_chain = SlitherRunnerOnChain()
                slither_findings_json_filename = slither_runner_on_chain.run_slither_onchain_scanner(smart_contract_address)
                if (slither_findings_json_filename and len(slither_findings_json_filename.strip()) > 0):
                    with open(slither_findings_json_filename, 'r') as f:
                        parsed_result = json.load(f)
                        normalized_score = util.calculate_risk_score(parsed_result)
                        response["risk_score"] = normalized_score
                        response["scan_result"] = parsed_result
                else:
                    response["ERROR"] = "Somthing went wrong. Data can't be saved. No file found."
            else:
                response["ERROR"] = "Invalid smart contract address. Please specify a valid & verified Polygon Mainnet smart contract address. See this - https://polygonscan.com/contractsVerified"

        elif chain_category == BlockchainCategory.POLYZKEVM:
            response["blockchain_category"] = BlockchainCategory.POLYZKEVM.name
            response["contract_address"] = smart_contract_address

            if network_id == PolygonZKEVMNetworkType.MAINNET:
                response["network_id"]=PolygonZKEVMNetworkType.MAINNET.name
            # elif network_id == PolygonZKEVMNetworkType.TESTNET:
            #     response["network_id"] = PolygonZKEVMNetworkType.TESTNET.name
            else:
                response["ERROR"] = "Invalid TESTNET /  MAINNET selection"

            smart_contract_analyzer = SmartContractAnalyzer()

            contract_adddress_hash = smart_contract_analyzer.generate_contract_hash(
                contract_address=response["contract_address"],
                blockchain_name=response["blockchain_category"],
                network_type=response["network_id"])

            db_data = db_handler.get_findings_from_db(contract_adddress_hash=contract_adddress_hash)

            if db_data is not None and "ERROR" not in db_data:
                if ("scan_result" in db_data):
                    result_summary = Util().calculate_findings_summary(db_data)
                    db_data["result_summary"] = result_summary
                    db_data["risk_score"] = round(db_data["risk_score"], 2)
                    db_data["scan_result"] = util.mask_scan_result_descriptions(db_data["scan_result"])
                    # del db_data["scan_result"]

                return db_data


            is_valid, is_verified = util.is_valid_verified_polygon_contract(smart_contract_address)
            if is_valid and is_verified:
                slither_runner_on_chain = SlitherRunnerOnChain()

                if not smart_contract_address.startswith("poly:"):
                    smart_contract_address = f"poly:{smart_contract_address}"

                slither_findings_json_filename = slither_runner_on_chain.run_slither_onchain_scanner(smart_contract_address)
                if (slither_findings_json_filename and len(slither_findings_json_filename.strip()) > 0):
                    with open(slither_findings_json_filename, 'r') as f:
                        parsed_result = json.load(f)
                        normalized_score = util.calculate_risk_score(parsed_result)
                        response["risk_score"] = normalized_score
                        response["scan_result"] = parsed_result
                else:
                    response["ERROR"] = "Somthing went wrong. Data can't be saved. No file found."
            else:
                response["ERROR"] = "Invalid smart contract address. Please specify a valid & verified Polygon Mainnet smart contract address. See this - https://polygonscan.com/contractsVerified"
        else:
            response["ERROR"] = "Supports ONLY ETHEREUM, POLYGON and ZKEVM MAINNET for now."

        if db_data is None or "ERROR" not in response:
            db_handler.save_findings_to_db(contract_adddress_hash=contract_adddress_hash,
                                                         json_output= response)
        if ("scan_result" in response):
            result_summary = Util().calculate_findings_summary(response)
            response["result_summary"] = result_summary
            response["scan_result"] = util.mask_scan_result_descriptions(response["scan_result"])
            # del response["scan_result"]
        return response

    except Exception as e:
        print("###############",e)
        return {"message": f"Error Occurred. {e}"}


class CodeReview(BaseModel):
    source_code: str
    file_name: str
    account_id: str

class Finding(BaseModel):
    fileName: str
    lineNumbers: str
    vulnerableCodeLines: str
    issueDescription: str
    CWE: str
    issuePriority: str

class Findings(BaseModel):
    source_code: str = ""
    findings : List[Finding] = []


if __name__ == "__main__":
    host = os.getenv("HOST")
    port = 8080
    uvicorn.run(app, host=host, port=port)

