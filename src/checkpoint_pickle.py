"""Resolve trusted script-origin checkpoints independently of Streamlit's __main__."""
from pickle import *  # noqa: F403 - torch's pickle-module interface
from pickle import Unpickler as StandardUnpickler

class Unpickler(StandardUnpickler):
    def find_class(self, module, name):
        if module == '__main__' and name == 'GraphFeatures':
            from revision import GraphFeatures
            return GraphFeatures
        if module == '__main__' and name == 'Features':
            from pipeline import Features
            return Features
        return super().find_class(module, name)
