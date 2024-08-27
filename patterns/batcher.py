import numpy as np


class Batcher:
    batch_size = 1000  # Default value, can be overridden in class definition
    prefetch_threshold = 700  # Calculated based on batch_size
    cache = {}
    init_method = set()
    atomic_outputs = {}
    remaining = 0
    prefetching = remaining < prefetch_threshold

    def __init__(self, init_method=None, atomic_outputs=None):
        if init_method is not None:
            assert len(Batcher.init_method.union(set([init_method]))) == 1, \
                "Only one init_method can be specified"
            Batcher.init_method = set([init_method,])
            # assert init_method can take batch_size argument
            assert "batch_size" in init_method.__code__.co_varnames, \
                "init_method must take batch_size argument"
        if atomic_outputs is not None:
            Batcher.atomic_outputs[atomic_outputs] = False
        Batcher.prefetching = Batcher.remaining < Batcher.prefetch_threshold

    def batching_init(self, func):
        def wrapper(*args, **kwargs):
            method_name = func.__name__
            if method_name not in self.cache:
                kwargs["batch_size"] = 1
                batch_result = func(*args, **kwargs)
                self.cache[method_name] = self._initialize_cache(batch_result)
            
            Batcher.prefetching = (Batcher.remaining < Batcher.prefetch_threshold)
            if Batcher.prefetching:
                kwargs["batch_size"] = Batcher.batch_size
                new_batch_result = func(*args, **kwargs)
                self._append_to_cache(method_name, new_batch_result)
            
            return self.cache[method_name]
        return wrapper

    def intermediate_batch(self, func):
        def wrapper(*args, **kwargs):
            method_name = func.__name__
            if method_name not in self.cache:
                batch_result = func(*args, **kwargs)
                self.cache[method_name] = self._initialize_cache(batch_result)
            
            if Batcher.prefetching:
                new_batch_result = func(*args, **kwargs)
                self._append_to_cache(method_name, new_batch_result)

            return self.cache[method_name]
        return wrapper
    

    def atomic_output(self, func):
        def wrapper(*args, **kwargs):
            method_name = func.__name__
            self.atomic_outputs[method_name] = True
            if method_name not in self.cache:
                batch_result = func(*args, **kwargs)
                self.cache[method_name] = self._initialize_cache(batch_result)
            
            finish_output = np.all(list(self.atomic_outputs.values()))    
            if Batcher.prefetching:
                new_batch_result = func(*args, **kwargs)
                self._append_to_cache(method_name, new_batch_result, increment_remaining=finish_output)
                Batcher.prefetching = (Batcher.remaining < Batcher.prefetch_threshold)

            atomic_item = self._extract_atomic_item(self.cache[method_name], decrement_remaining=finish_output)
            
            if finish_output:
                for k in self.atomic_outputs.keys():
                    self.atomic_output[k] = False
            return atomic_item
        return wrapper
    
    def _initialize_cache(self, data):
        if isinstance(data, tuple):
            return [np.array([]) for item in data]
        else:
            return np.array([])

    def _extract_atomic_item(self, cache, atomic_item, decrement_remaining=False):
        if isinstance(cache, tuple) and all(isinstance(item, np.ndarray) for item in cache):
            # If the cache is a tuple of numpy arrays
            atomic_item = tuple(item[0] for item in cache)  # Extract the first element from each array
        elif isinstance(cache, np.ndarray):
            # If the cache is a single numpy array
            atomic_item = cache[0]
        else:
            raise TypeError("Cache format not recognized")
        # update all caches
        for method in self.cache.keys():
            if isinstance(self.cache[method], tuple) and all(isinstance(item, np.ndarray) for item in self.cache[method]):
                self.cache[method] = tuple(item[1:] for item in self.cache[method])
            elif isinstance(self.cache[method], np.ndarray):
                self.cache[method] = self.cache[method][1:]
                
        if decrement_remaining:
            Batcher.remaining -= 1
        if Batcher.prefetching:
            Batcher.prefetching = (Batcher.remaining < Batcher.prefetch_threshold)
        return atomic_item
        

    def _append_to_cache(self, method_name, new_batch, increment_remaining=False):
        if isinstance(new_batch, tuple):
            for i in range(len(new_batch)):
                self.cache[method_name][i] = np.concatenate((self.cache[method_name][i], new_batch[i]), axis=0)
        else:
            self.cache[method_name] = np.concatenate((self.cache[method_name], new_batch), axis=0)

        if increment_remaining:
            Batcher.remaining += len(self.batch_size)
