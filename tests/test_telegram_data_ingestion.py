import unittest
import json
import pandas as pd

class TestTelegramDataIngestion(unittest.TestCase):

    def test_json_loading(self):
        # Test if the JSON file loads correctly
        with open('telegram_messages.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIsInstance(data, list)  # Ensure it's a list

    def test_data_preservation(self):
        # Test if the data is preserved in the expected format
        df = pd.read_json('telegram_messages.json')
        self.assertIn('content', df.columns)  # Check if 'content' column exists
        self.assertTrue(df['content'].notnull().all())  # Ensure no null values

    def test_csv_output(self):
        # Test if the CSV file is generated correctly
        df = pd.read_json('telegram_messages.json')
        df.to_csv('preprocessed_data.csv', index=False, encoding='utf-8-sig')

        # Check if CSV file exists and can be read
        df_csv = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')
        self.assertEqual(len(df), len(df_csv))  # Ensure the number of rows match

if __name__ == '__main__':
    unittest.main()