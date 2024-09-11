import unittest


class UnitTests(unittest.TestCase):
    def test_example(self):
        print("[INFO] Example test")
        result = True
        self.assertEqual(True, result)


if __name__ == "__main__":
    unittest.main()
