import unittest
from config.config_loader import ConfigLoader

@ConfigLoader.register_class
class SubClass:
    def __init__(self, param3):
        self.param3 = param3

@ConfigLoader.register_class
class MyClass:
    def __init__(self, param1, param2, sub_obj):
        self.param1 = param1
        self.param2 = param2
        self.sub_obj = sub_obj


class TestConfigLoader(unittest.TestCase):
    def test_create_object(self):
        # Define the YAML configuration for testing
        config_data = {
            "my_class": {
                "class_name": "MyClass",
                "params": {
                    "param1": "value1",
                    "param2": "value2",
                    "sub_obj": {
                        "class_name": "SubClass",
                        "params": {
                            "param3": "value3"    
                        }
                    }
                }
            },
        }

        # Create an instance of ConfigLoader
        loader = ConfigLoader(config_data=config_data)

        # Create an object of MyClass using the YAML configuration
        my_class_obj = loader.create_or_get_object("MyClass", config_data["my_class"])

        # Assert the values of the object's attributes
        self.assertEqual(my_class_obj.param1, "value1")
        self.assertEqual(my_class_obj.param2, "value2")
        self.assertEqual(my_class_obj.sub_obj.param3, "value3")
        
    def test_create_object_ref(self):
        # Define the YAML configuration for testing
        config_data = {
            "my_class": {
                "class_name": "MyClass",
                "params": {
                    "param1": "value1",
                    "param2": "value2",
                    "sub_obj": {
                        "ref": "shared_obj"
                    }
                }
            },
            "shared_obj": {
                "class_name": "SubClass",
                "params": {
                    "param3": "value3"    
                }
            }
        }

        # Create an instance of ConfigLoader
        loader = ConfigLoader(config_data=config_data)

        # Create an object of MyClass using the YAML configuration
        my_class_obj = loader.create_or_get_object("MyClass", config_data["my_class"])

        # Assert the values of the object's attributes
        self.assertEqual(my_class_obj.param1, "value1")
        self.assertEqual(my_class_obj.param2, "value2")
        self.assertEqual(my_class_obj.sub_obj.param3, "value3")

    def test_argparser(self):
        # Define the YAML configuration for testing
        config_data = {
            "my_class": {
                "class_name": "MyClass",
                "params": {
                    "param1": "value1",
                    "param2": "value2",
                    "sub_obj": {
                        "ref": "shared_obj"
                    }
                }
            },
            "shared_obj": {
                "class_name": "SubClass",
                "params": {
                    "param3": "value3"    
                }
            }
        }


        # Create an instance of ConfigLoader
        loader = ConfigLoader(config_data=config_data, cmd_args="--my_class.sub_obj.param3=0.2 --my_class.param1=[1, 2, 3]")
        loader.load_objects()
        # Assert that the value was updated correctly
        self.assertEqual(loader['shared_obj'].param3, 0.2)
        self.assertEqual(loader['my_class'].param1, [1, 2, 3])

if __name__ == "__main__":
    unittest.main()