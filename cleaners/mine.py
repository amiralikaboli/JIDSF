import json
import re
from collections import defaultdict

import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


class Cleaner:
    def __init__(self, db_path):
        self.replace_invalid_chars = {
            "": ['"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '-', '/', ':', '<', '=', '>', '@', '\\', '`', '~',
                 '’'],
            " ": [',', '.', '!', '?']
        }

        self.num_pattern = re.compile("[0-9]+")
        self.clock12_pattern = re.compile("(1[012]|[1-9]):[0-5][0-9](\\s)?(?i)(am|pm)")
        self.clock24_pattern = re.compile("([01]?[0-9]|2[0-3]):[0-5][0-9]")

        self.db_items, self.db_regexes = self.build_db_regexes(db_path)
        self.db_patterns = {db_name: re.compile(db_regex) for db_name, db_regex in self.db_regexes.items()}

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def build_db_regexes(self, db_path):  # NEW CHANGES
        items = defaultdict(set)
        regexes = defaultdict(str)

        for db_name in ["attraction", "hotel", "restaurant"]:
            with open(f"{db_path}/{db_name}_db.json", "r") as json_file:
                db = json.load(json_file)
                regexes[db_name] = "|".join(set(f'({db_obj["name"]})' for db_obj in db))
                items["phone"].update([db_obj["phone"] for db_obj in db if "phone" in db_obj.keys()])
                items["postcode"].update([db_obj["postcode"] for db_obj in db])

        with open(f"{db_path}/hospital_db.json", "r") as json_file:
            db = json.load(json_file)
            regexes["hospital"] = "|".join(set(f'({db_obj["department"]})' for db_obj in db))
            items["phone"].update([db_obj["phone"] for db_obj in db])

        with open(f"{db_path}/train_db.json", "r") as json_file:
            db = json.load(json_file)
            items["trainID"] = set(db_obj["trainID"] for db_obj in db)

        for key in items.keys():
            # print(key, len(items[key]))
            regexes[key] = "|".join([f"({elem})" for elem in items[key]])

        return items, regexes

    def replace_num(self, text):
        text = self.num_pattern.sub("NUM", text)
        return text

    def replace_clock(self, text):
        text = self.clock12_pattern.sub("CLOCK", text)
        text = self.clock24_pattern.sub("CLOCK", text)
        return text

    def replace_dbs(self, text):
        words = text.split()
        for db_name in self.db_regexes.keys():
            if db_name not in self.db_items.keys() or self.db_items[db_name].intersection(words):
                text = self.db_patterns[db_name].sub(db_name.upper(), text)
        return text

    def clean(self, text):
        text = text.lower()
        text = contractions.fix(text)
        text = self.replace_dbs(text)
        text = self.replace_clock(text)
        for replace_char, invalid_chars in self.replace_invalid_chars.items():
            for invalid_char in invalid_chars:
                text = text.replace(invalid_char, replace_char)
        text = self.replace_num(text)
        return text

    def tokenize(self, text):
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) if not word.isupper() else word for word in words]
        words = [self.stemmer.stem(word) if not word.isupper() else word for word in words]
        words = [word for word in words if word not in self.stop_words]
        return words
