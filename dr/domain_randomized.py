from dr.generator import DRGenerator


class DomainRandomized:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.random_params = {}
        self.static_params = {}
        self.classify_params()
    
    def classify_params(self):
        for key, value in self.params.items():
            if isinstance(value, DRGenerator):
                self.random_params[key] = value
            else:
                self.static_params[key] = value
    
    def initialize(self):
        initial_values = {}
        for key, value in self.static_params.items():
            initial_values[key] = value
        for key, value in self.random_params.items():
            initial_values[key] = value.generate()
        return initial_values
