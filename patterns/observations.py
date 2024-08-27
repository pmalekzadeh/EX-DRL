"""Observer and Observable Pattern"""
from abc import ABC

class Observable(ABC):
    def __init__(self):
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.insert(0, observer)
    
    def remove_observer(self, observer):
        self.observers.remove(observer)

    def notify_observers(self, data={}):
        for observer in self.observers:
            observer.update(data)

class Observer(ABC):
    def update(self, data):
        pass
