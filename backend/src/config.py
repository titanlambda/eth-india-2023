import os

from dotenv import dotenv_values, load_dotenv, find_dotenv

class Config(object):
    def __init__(self):
        dotenv_path_in_proj = find_dotenv()
        dot_config = dotenv_values(dotenv_path=dotenv_path_in_proj)
        self._config = dot_config

    def get_property(self, property_name):
        if property_name not in self._config.keys():  # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]


class PolygonScanConfig(Config):
    @property
    def polygon_scan_api_key(self):
        return self.get_property("POLYSCAN_API_KEY")

class PolygonScanZKEVMConfig(Config):
    @property
    def polygon_scan_ZKEVM_api_key(self):
        return self.get_property("POLYSCAN_ZKEVM_API_KEY")

class EtherScanConfig(Config):
    @property
    def ether_scan_api_key(self):
        return self.get_property("ETHEREUM_API_KEY")

