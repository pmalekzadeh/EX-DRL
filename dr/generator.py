from scipy import stats

from config.config_loader import ConfigLoader


@ConfigLoader.register_class
class DRGenerator(object):
    def __init__(self, distribution='uniform', **kwargs):
        self.distribution_name = distribution
        self.params = kwargs
        self.distribution = self._create_distribution()

    def _create_distribution(self):
        if hasattr(stats, self.distribution_name):
            return getattr(stats, self.distribution_name)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution_name}")

    def generate(self):
        return self.distribution.rvs(**self.params)