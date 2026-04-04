import unittest

from train_fasttext import tokenize


class TestTokenization(unittest.TestCase):

    def test_tokenization_with_mixed_content(self):
        input = "hello 你好 123"
        expected = ['hello', '你', '好', '123']

        actual = tokenize(input)
        self.assertEqual(actual, expected)

    def test_tokenization_with_punctuation_and_parentheses(self):
        input = "你好！(abc)"
        expected = ['你', '好', '！', '(', 'abc', ')']

        actual = tokenize(input)
        self.assertEqual(actual, expected)

    def test_tokenization_with_only_chinese_characters(self):
        input = "世界和平"
        expected = ['世', '界', '和', '平']

        actual = tokenize(input)
        self.assertEqual(actual, expected)

    def test_tokenization_with_only_english_characters(self):
        input = "hello world"
        expected = ['hello', 'world']

        actual = tokenize(input)
        self.assertEqual(actual, expected)

    def test_tokenization_with_only_numbers(self):
        input = "123"
        expected = ['123']

        actual = tokenize(input)
        self.assertEqual(actual, expected)

    def test_tokenization_with_empty_string(self):
        input = ""
        expected = []

        actual = tokenize(input)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    # run all the tests defined in this class
    unittest.main()
