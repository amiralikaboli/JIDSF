import json
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


class Cleaner:
    def __init__(self, mapping_path, db_path):
        with open(mapping_path, "r") as txt_file:
            self.replacements = []
            for line in txt_file.readlines():
                tok_from, tok_to = line.replace('\n', '').split('\t')
                self.replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

        self.timepat = re.compile("\d{1,2}[:]\d{1,2}")
        self.pricepat = re.compile("\d{1,3}[.]\d{1,2}")
        self.digitpat = re.compile('\d+')

        self.dic = self.prepare_slot_values_independent(db_path)

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def prepare_slot_values_independent(self, db_path):
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']
        dic = []
        dic_area = []
        dic_food = []
        dic_price = []

        # read databases
        for domain in domains:
            try:
                with open(f'{db_path}/{domain}_db.json') as json_file:
                    db_json = json.load(json_file)

                for ent in db_json:
                    for key, val in ent.items():
                        if val == '?' or val == 'free':
                            pass
                        elif key == 'address':
                            dic.append((self.normalize(val), '[' + domain + '_' + 'address' + ']'))
                            if "road" in val:
                                val = val.replace("road", "rd")
                                dic.append((self.normalize(val), '[' + domain + '_' + 'address' + ']'))
                            elif "rd" in val:
                                val = val.replace("rd", "road")
                                dic.append((self.normalize(val), '[' + domain + '_' + 'address' + ']'))
                            elif "st" in val:
                                val = val.replace("st", "street")
                                dic.append((self.normalize(val), '[' + domain + '_' + 'address' + ']'))
                            elif "street" in val:
                                val = val.replace("street", "st")
                                dic.append((self.normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif key == 'name':
                            dic.append((self.normalize(val), '[' + domain + '_' + 'name' + ']'))
                            if "b & b" in val:
                                val = val.replace("b & b", "bed and breakfast")
                                dic.append((self.normalize(val), '[' + domain + '_' + 'name' + ']'))
                            elif "bed and breakfast" in val:
                                val = val.replace("bed and breakfast", "b & b")
                                dic.append((self.normalize(val), '[' + domain + '_' + 'name' + ']'))
                            elif "hotel" in val and 'gonville' not in val:
                                val = val.replace("hotel", "")
                                dic.append((self.normalize(val), '[' + domain + '_' + 'name' + ']'))
                            elif "restaurant" in val:
                                val = val.replace("restaurant", "")
                                dic.append((self.normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif key == 'postcode':
                            dic.append((self.normalize(val), '[' + domain + '_' + 'postcode' + ']'))
                        elif key == 'phone':
                            dic.append((val, '[' + domain + '_' + 'phone' + ']'))
                        elif key == 'trainID':
                            dic.append((self.normalize(val), '[' + domain + '_' + 'id' + ']'))
                        elif key == 'department':
                            dic.append((self.normalize(val), '[' + domain + '_' + 'department' + ']'))

                        # NORMAL DELEX
                        elif key == 'area':
                            dic_area.append((self.normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                        elif key == 'food':
                            dic_food.append((self.normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                        elif key == 'pricerange':
                            dic_price.append((self.normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                        else:
                            pass
                        # TODO car type?
            except:
                pass

            if domain == 'hospital':
                dic.append((self.normalize('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
                dic.append((self.normalize('Hills Road'), '[' + domain + '_' + 'address' + ']'))
                dic.append((self.normalize('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
                dic.append(('01223245151', '[' + domain + '_' + 'phone' + ']'))
                dic.append(('1223245151', '[' + domain + '_' + 'phone' + ']'))
                dic.append(('0122324515', '[' + domain + '_' + 'phone' + ']'))
                dic.append((self.normalize('Addenbrookes Hospital'), '[' + domain + '_' + 'name' + ']'))

            elif domain == 'police':
                dic.append((self.normalize('Parkside'), '[' + domain + '_' + 'address' + ']'))
                dic.append((self.normalize('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
                dic.append(('01223358966', '[' + domain + '_' + 'phone' + ']'))
                dic.append(('1223358966', '[' + domain + '_' + 'phone' + ']'))
                dic.append((self.normalize('Parkside Police Station'), '[' + domain + '_' + 'name' + ']'))

        # add at the end places from trains
        with open(f'{db_path}/train_db.json') as json_file:
            db_json = json.load(json_file)

        for ent in db_json:
            for key, val in ent.items():
                if key == 'departure' or key == 'destination':
                    dic.append((self.normalize(val), '[' + 'value' + '_' + 'place' + ']'))

        # add specific values:
        for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            dic.append((self.normalize(key), '[' + 'value' + '_' + 'day' + ']'))

        # more general values add at the end
        dic.extend(dic_area)
        dic.extend(dic_food)
        dic.extend(dic_price)

        return dic

    def insert_space(self, token, text):
        sidx = 0
        while True:
            sidx = text.find(token, sidx)
            if sidx == -1:
                break
            if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                    re.match('[0-9]', text[sidx + 1]):
                sidx += 1
                continue
            if text[sidx - 1] != ' ':
                text = text[:sidx] + ' ' + text[sidx:]
                sidx += 1
            if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                text = text[:sidx + 1] + ' ' + text[sidx + 1:]
            sidx += 1
        return text

    def normalize(self, text):
        # lower case every word
        text = text.lower()

        # replace white spaces in front and end
        text = re.sub(r'^\s*|\s*$', '', text)

        # hotel domain pfb30
        text = re.sub(r"b&b", "bed and breakfast", text)
        text = re.sub(r"b and b", "bed and breakfast", text)

        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # normalize postcode
        ms = re.findall(
            '([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
            text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

        # weird unicode bug
        text = re.sub(u"(\u2018|\u2019)", "'", text)

        # replace time and and price
        text = re.sub(self.timepat, ' [value_time] ', text)
        text = re.sub(self.pricepat, ' [value_price] ', text)
        # text = re.sub(pricepat2, '[value_price]', text)

        # replace st.
        text = text.replace(';', ',')
        text = re.sub('$\/', '', text)
        text = text.replace('/', ' and ')

        # replace other special characters
        text = text.replace('-', ' ')
        text = re.sub('[\":\<>@\(\)]', '', text)

        # insert white space before and after tokens:
        for token in ['?', '.', ',', '!']:
            text = self.insert_space(token, text)

        # insert white space for 's
        text = self.insert_space('\'s', text)

        # replace it's, does't, you'd ... etc
        text = re.sub('^\'', '', text)
        text = re.sub('\'$', '', text)
        text = re.sub('\'\s', ' ', text)
        text = re.sub('\s\'', ' ', text)
        for fromx, tox in self.replacements:
            text = ' ' + text + ' '
            text = text.replace(fromx, tox)[1:-1]

        # remove multiple spaces
        text = re.sub(' +', ' ', text)

        # concatenate numbers
        tmp = text
        tokens = text.split()
        i = 1
        while i < len(tokens):
            if re.match(u'^\d+$', tokens[i]) and \
                    re.match(u'\d+$', tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)

        return text

    def delexicalise(self, utt):
        for key, val in self.dic:
            utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
            utt = utt[1:-1]  # why this?
        utt = " ".join(utt.split())
        return utt

    def delexicalise_reference_number(self, sent, turn):
        """Based on the belief state, we can find reference number that
        during data gathering was created randomly."""
        if turn is None:
            return sent

        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
        if turn['metadata']:
            for domain in domains:
                if turn['metadata'][domain]['book']['booked']:
                    for slot in turn['metadata'][domain]['book']['booked'][0]:
                        if slot == 'reference':
                            val = '[' + domain + '_' + slot + ']'
                        else:
                            val = '[' + domain + '_' + slot + ']'
                        key = self.normalize(turn['metadata'][domain]['book']['booked'][0][slot])
                        sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                        # try reference with hashtag
                        key = self.normalize("#" + turn['metadata'][domain]['book']['booked'][0][slot])
                        sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                        # try reference with ref#
                        key = self.normalize("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
                        sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        sent = " ".join(sent.split())
        return sent

    def clean(self, text, turn=None):
        sent = self.normalize(text)
        sent = self.delexicalise(sent)
        #         sent = self.delexicalise_reference_number(sent, turn)
        sent = re.sub(self.digitpat, '[value_count]', sent)
        return sent

    def tokenize(self, text):
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) if not word.isupper() else word for word in words]
        # words = [self.stemmer.stem(word) if not word.isupper() else word for word in words]
        # words = [word for word in words if word not in self.stop_words]
        return words
