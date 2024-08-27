'''LazyObject pattern

Use this pattern to create objects that are only evaluated once and then stored.
This is useful for expensive calculations that are only needed once, such as option pricing.

Example:
class Option(LazyBase):
    @lazy_func
    def price_and_greek(self):
        # Your code here...
        pass

'''

class LazyObject:
    def __init__(self, func, instance):
        self.func = func
        self.instance = instance
        self.value = None
        self.evaluated = False

    def evaluate(self):
        if not self.evaluated:
            self.value = self.func(self.instance)
            self.evaluated = True
        return self.value

    def reset(self):
        self.evaluated = False

class LazyBase:
    def __init__(self):
        self._lazy_objects = {}

    def clear(self):
        """
        Resets all lazy objects in the `_lazy_objects` dictionary.
        """
        for lazy_object in self._lazy_objects.values():
            lazy_object.reset()
    
    @staticmethod
    def lazy_func(func):
        def wrapper(self):
            if not hasattr(self, '_lazy_objects'):
                self._lazy_objects = {}
            if func.__name__ not in self._lazy_objects:
                self._lazy_objects[func.__name__] = LazyObject(func, self)
            return self._lazy_objects[func.__name__].evaluate()
        return wrapper


