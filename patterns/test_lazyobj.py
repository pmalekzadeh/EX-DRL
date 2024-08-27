import unittest

from patterns.lazyobj import LazyBase

class TestLazyObject(unittest.TestCase):
    def test_lazy_object(self):
        class TestClass(LazyBase):
            @LazyBase.lazy_func
            def expensive_computation(self):
                return 42

        obj = TestClass()

        # Test that the computation gives the correct result
        self.assertEqual(obj.expensive_computation(), 42)
        
        # Test that the computation has been done
        self.assertTrue(obj._lazy_objects['expensive_computation'].evaluated)

        # Test that the result is cached
        self.assertEqual(obj.expensive_computation(), 42)

        # Test that clear resets the computation
        obj.clear()
        self.assertFalse(obj._lazy_objects['expensive_computation'].evaluated)

if __name__ == '__main__':
    unittest.main()