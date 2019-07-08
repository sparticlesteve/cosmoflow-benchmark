"""Utility code for argparse"""

import argparse
import yaml

#class StoreDictKeyPair(argparse.Action):
#    """An action for reading key-value pairs from command line"""
#    def __call__(self, parser, namespace, values, option_string=None):
#        my_dict = {}
#        for kv in values.split(","):
#            k,v = kv.split("=")
#            my_dict[k] = v
#        setattr(namespace, self.dest, my_dict)

class ReadYaml(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = yaml.load(values, Loader=yaml.FullLoader)
        setattr(namespace, self.dest, my_dict)
