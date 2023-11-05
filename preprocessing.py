import pandas as pd
import nltk
import string
import re
import json
import datetime
from collections import OrderedDict
from check_instructions import check_instructions
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from utils import emoticons
from utils import emojis_unicode
from utils import slang_words
from spellchecker import SpellChecker
from dateutil import parser

#-------------------------------------------------------------Instructions--------------------------------------------------------------
"""
You have to create a json file and give the path of this file as an argument to the preprocessing function. The second argument is the dataframe that you want to preprocess.
The json file must contain the instructions to preprocess the text and must be written as follows:

The keys of the dictionary are the names of the columns to preprocess. If you want to preprocess all the columns, you can write "ALL" as a key. If you want to preprocess several columns with the same instructions, you can write "COLUMNS;column1;column2;..." as a key. If you want to preprocess only one column, you can write the name of the column as a key.
These keys have for value a dictionary that contains the instructions to preprocess the column. The keys of this dictionary are the instructions to apply to the column. The values of this dictionary are dictionaries that contain the arguments of the instructions and their value.

Here are the instructions that you can use and their arguments:

LOWERCASE: lowercases the text (no argument)
REMOVE_PUNCT: removes punctuation (takes an optional argument punctuation which is a string containing the punctuation to remove. In case no argument is given, the function removes all the punctuation)
REMOVE_STOPWORDS: removes stopwords (takes an optional argument language which is a string containing the language of the stopwords to remove. In case no argument is given, the function removes english stopwords)
REMOVE_FREQUENT: removes the most frequent words (takes one or two arguments. The argument nb_words is mandatory and is the number of most frequent words to remove. The argument apply is optional and is a string containing -e if you want to apply the removal of the frequent words. If there is no argument apply, the function only prints the number of most frequent words indicated in the argument nb_words without removing them)
REMOVE_RARE: removes the most rare words (takes one or two arguments. The argument nb_words is mandatory and is the number of most rare words to remove. The argument apply is optional and is a string containing -e if you want to apply the removal of the rare words. If there is no argument apply, the function only prints the number of most rare words indicated in the argument nb_words without removing them)
STEM: stems the words (takes an optional argument language which is a string containing the language of the words to stem. In case no argument is given, the function stems english words)
LEMMATIZE_ENGLISH: lemmatizes the words (no argument)
REMOVE_EMOJI: removes emojis (no argument)
REMOVE_EMOTICONS: removes emoticons (no argument)
CONVERT_EMOJIS: converts emojis to words (no argument)
CONVERT_EMOTICONS: converts emoticons to words (no argument)
REMOVE_URLS: removes urls (no argument)
REMOVE_HTML: removes html (no argument)
CHAT_WORDS_CONVERSION: converts chat words to words (no argument)
SPELL_CORRECTION: corrects spelling (no argument)
CONVERT_TO_DATE_OR_DATETIME: converts the text to date or datetime (no argument)
EXTRACT_REGEX_PATTERN: extracts a regex pattern (takes one to four arguments. The argument regex_pattern is mandatory and is the regex pattern to extract. The argument secondary_regex_pattern is optional and is the secondary regex pattern to extract in the first regex_pattern. The argument new_column_name is optional and is the name of the new column(s) that will contain the result of the extraction. If there is no argument new_column_name the result will replace the column(s) that was(were) extracted. The argument result_type is optional and is the type of the result of the extraction. If there is no argument result_type, the result is a list.)

It's possible to use multiple times the same instruction on the same object if you put "_{number}" at the end of the instruction. For example, if you want to apply the instruction EXTRACT_REGEX_PATTERN twice, you can write EXTRACT_REGEX_PATTERN and EXTRACT_REGEX_PATTERN_1.

structure of the instructions with all the arguments in the json file:

{"ALL":{
    "LOWERCASE": {},
    "REMOVE_PUNCT": {"punctuation": "!?,"},
    "REMOVE_STOPWORDS": {"language": "french"},
    "REMOVE_FREQUENT": {"nb_words": 10, "apply": "-e"},
    "REMOVE_RARE": {"nb_words": 10, "apply": "-e"},
    "STEM": {"language": "english"},
    "LEMMATIZE_ENGLISH": {},
    "REMOVE_EMOJI": {},
    "REMOVE_EMOTICONS": {},
    "CONVERT_EMOJIS": {},
    "CONVERT_EMOTICONS": {},
    "REMOVE_URLS": {},
    "REMOVE_HTML": {},
    "CHAT_WORDS_CONVERSION": {},
    "SPELL_CORRECTION": {},
    "CONVERT_TO_DATE_OR_DATETIME": {},
    "EXTRACT_REGEX_PATTERN": {"regex_pattern": "regex", "secondary_regex_pattern": "regex", "new_column_name": "new_column", "result_type": "str"}
}
}
"""


#-----------------------------------------------------Dict to check instructions--------------------------------------------------------
dict_check_instructions = {"LOWERCASE": {},
                            "REMOVE_PUNCT": {"punctuation": ["str"]},
                            "REMOVE_STOPWORDS": {"language": ["str_inlist", stopwords.fileids()]},
                            "REMOVE_FREQUENT": {"nb_words": ["int", "mandatory"], "apply": ["str_inlist", ["-e"]]},
                            "REMOVE_RARE": {"nb_words": ["int", "mandatory"], "apply": ["str_inlist", ["-e"]]},
                            "STEM": {"language": ["str_inlist", SnowballStemmer.languages]},
                            "LEMMATIZE_ENGLISH": {},
                            "REMOVE_EMOJI": {},
                            "REMOVE_EMOTICONS": {},
                            "CONVERT_EMOJIS": {},
                            "CONVERT_EMOTICONS": {},
                            "REMOVE_URLS": {},
                            "REMOVE_HTML": {},
                            "CHAT_WORDS_CONVERSION": {},
                            "SPELL_CORRECTION": {},
                            "CONVERT_TO_DATE_OR_DATETIME": {},
                            "EXTRACT_REGEX_PATTERN": {"regex_pattern": ["str", "mandatory"], "secondary_regex_pattern": ["str"],  "new_column_name": ["list"], "result_type": ["str_inlist", ["list", "str", "int", "float", "date", "datetime"]]}
                            }

#-----------------------------------------------------Functions for preprocessing-------------------------------------------------------

# lowercasing
def lowercase(text):
    text = str(text)
    return text.lower()

# remove the punctuation given in argument
def remove_punctuation(text, punctuation):
    text = str(text)
    translation_table = str.maketrans('', '', punctuation)
    return text.translate(translation_table)

# remove stopwords given in argument
def remove_stopwords(text, STOPWORDS):
    text = str(text)
    text_split = text.split(' ')
    text_filtered =  [word for word in text_split if word   not in STOPWORDS]
    text_filtered = " ".join(text_filtered)
    return text_filtered

# remove frequent words or rare words depending on the argument freq_words
def remove_freq_or_rare_words(text, freq_words):
    text = str(text)
    text_removed = " ".join([word for word in text.split(" ") if word not in freq_words])
    return text_removed

# stem the words according to the language given in argument
def stem_words(text, language):
    text = str(text)
    stemmer = SnowballStemmer(language)
    text_stemmed = " ".join([stemmer.stem(word) for word in text.split(" ")])
    return text_stemmed

# lemmatize the words in english
def lemmatize_words_eng(text):
    text = str(text)

    lemmatizer = WordNetLemmatizer()
    wordnet_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }

    pos_tagged_text = nltk.pos_tag(nltk.tokenize.word_tokenize(text))
    lemmatized_words = [lemmatizer.lemmatize(words, pos = wordnet_map.get(pos[0], wordnet.NOUN)) for words, pos in pos_tagged_text ]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# remove emojis
def remove_emoji(text):
    text = str(text)
    emoji_pattern = re.compile("[" 
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"  # Miscellaneous symbols
                           u"\U000024C2-\U0001F251"  # Enclosed characters
                           "]+", flags=re.UNICODE)   # '+' signifies that those characters can occur once or more consecutively

    text_no_emoji = re.sub(emoji_pattern, '', text)
    return text_no_emoji

# remove emoticons
def remove_emoticons(text):
    text = str(text)
    EMOTICONS = emoticons()
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

# convert emojis to words
def convert_emojis(text):
    text = str(text)
    EMO_UNICODE = emojis_unicode()
    for description, emoji in EMO_UNICODE.items():
        description = description.replace(":", "")
        description = description.replace(",", "").split(" ")
        description = "_".join(description)
        text = text.replace(emoji, description)
    return text

# convert emoticons to words
def convert_emoticons(text):
    text = str(text)
    EMOTICONS = emoticons()
    for emoticon, description in EMOTICONS.items():
        description = description.replace(",", "").split(" ")
        description = "_".join(description)
        text = re.sub(emoticon, description, text)
    return text

# remove urls
def remove_urls(text):
    text = str(text)
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# remove html
def remove_html(text):
    text = str(text)
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# convert chat words to words
def chat_words_conversion(text):
    text = str(text)
    slang_words_list = slang_words()
    chat_words_list = list(slang_words_list.keys())
    new_text = []
    for word in text.split(" "):
        if word.upper() in chat_words_list:
            new_text.append(slang_words_list[word.upper()])
        else:
            new_text.append(word)   
    new_text = " ".join(new_text)
    return new_text

# correct spelling mistakes
def correct_spellings(text):
    text = str(text)
    spell = SpellChecker()
    corrected_text = []
    misspelled = spell.unknown(text.split(" "))
    for word in text.split(" "):
        if word == " ":
            continue
        if word in misspelled:
            if spell.correction(word) == None:
                corrected_text.append(word)
            else:
                corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    corrected_text = " ".join(corrected_text)
    return corrected_text

# convert to date or datetime
def convert_to_date_or_datetime(text):
    text = str(text)
    try:
        datetime_obj = parser.parse(text, fuzzy=True)
        if datetime_obj.time() != datetime.time(0):
            return datetime_obj
        else:
            return datetime_obj.date()
    except ValueError:
        return None

# extract regex pattern
def extract_regex_pattern(text, regex_pattern, secondary_regex_pattern, result_type):
    text = str(text)
    primary_matches = re.findall(regex_pattern, text)
    
    if secondary_regex_pattern:
        result = []
        for match in primary_matches:
            secondary_matches = re.findall(secondary_regex_pattern, match)
            for secondary_match in secondary_matches:
                result.append(secondary_match)
    else:
        result = primary_matches

    if not result:
        if result_type == "int":
            return 0
        elif result_type == "float":
            return 0.0
        elif result_type in ["date", "datetime", "str", "list"]:
            return None
        else:
            return None
    else:
        if result_type == "list":
            return result
        elif result_type == "str":
            return " ".join(result)
        elif result_type == "int":
            return int(result[0])
        elif result_type == "float":
            return float(result[0])
        elif result_type == "date":
            return parser.parse(result[0], fuzzy=True).date()
        elif result_type == "datetime":
            return parser.parse(result[0], fuzzy=True)
        else:
            return result


#-------------------------------------------------------------Preprocessing-------------------------------------------------------------
instructions_without_argument = {
    "LOWERCASE": lowercase,
    "REMOVE_EMOJI": remove_emoji,
    "REMOVE_EMOTICONS": remove_emoticons,
    "CONVERT_EMOJIS": convert_emojis,
    "CONVERT_EMOTICONS": convert_emoticons,
    "REMOVE_URLS": remove_urls,
    "REMOVE_HTML": remove_html,
    "CHAT_WORDS_CONVERSION": chat_words_conversion,
    "SPELL_CORRECTION": correct_spellings,
    "CONVERT_TO_DATE_OR_DATETIME": convert_to_date_or_datetime
}

def preprocessing(instructions_file, df):

    # check the instructions
    check_instructions(dict_check_instructions, instructions_file, df)  

    df_copy = df.copy()
    
    with open(instructions_file, 'r') as json_file:
        dict_json = json.load(json_file, object_pairs_hook=OrderedDict)
    # browse through the names of columns to preprocess
    for target, value in dict_json.items():
        if target == "ALL":
            columns_to_process = df_copy.columns
        elif target.startswith("COLUMNS"):
            list_columns = target.split(";")[1:]
            columns_to_process = list_columns
        else:
            columns_to_process = [target]

        dict_target = dict_json[target]
        # browse through the instructions of the column
        for instruction, value in dict_target.items():
            print(f"Instruction {instruction} in progress...")
            dict_instruction = dict_target[instruction]
            # check if it's an instruction with a number (ex: EXTRACT_REGEX_PATTERN_1) if we have to apply the same instruction several times 
            if instruction not in dict_check_instructions.keys():
                instruction = instruction[:-2]
            # removing punctuation
            if instruction == "REMOVE_PUNCT":
                if "punctuation" not in dict_instruction.keys():
                    df_copy[columns_to_process] = df_copy[columns_to_process].applymap(lambda line: remove_punctuation(line, string.punctuation))
                else:
                    df_copy[columns_to_process] = df_copy[columns_to_process].applymap(lambda line: remove_punctuation(line, dict_instruction["punctuation"]))
                continue
            # removing stopwords
            elif instruction == "REMOVE_STOPWORDS":
                nltk.download('stopwords', quiet=True)
                if "language" not in dict_instruction.keys():
                    STOPWORDS = set(stopwords.words('english'))
                else:
                    STOPWORDS = set(stopwords.words(dict_instruction["language"]))

                df_copy[columns_to_process] = df_copy[columns_to_process].applymap(lambda line: remove_stopwords(line, STOPWORDS))
                continue
            # lemmatize_english
            elif instruction == "LEMMATIZE_ENGLISH":
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('punkt', quiet=True)
                df_copy[columns_to_process] = df_copy[columns_to_process].applymap(lemmatize_words_eng)
                continue
            # removing frequent words or rare words
            elif instruction == "REMOVE_FREQUENT" or instruction == "REMOVE_RARE":
                for column in columns_to_process:
                    text_join = " ".join(df_copy[column])
                    text_split = text_join.split(" ")
                    text_split = [word for word in text_split if word.strip()]
                    count_words = Counter(text_split)
                    most_common = count_words.most_common()

                    if instruction == "REMOVE_FREQUENT":
                        if "nb_words" in dict_instruction.keys() and "apply" not in dict_instruction.keys():
                            print(f"Most frequent words in the column {column}\n{most_common[:int(dict_instruction['nb_words'])]}")
                        if "nb_words" in dict_instruction.keys() and "apply" in dict_instruction.keys():
                            print(f"Most frequent words in the column {column}\n{most_common[:int(dict_instruction['nb_words'])]}")
                            words_to_remove = [w for (w, word_count) in most_common[:int(dict_instruction["nb_words"])]]
                            df_copy[column] = df_copy[column].apply(lambda line: remove_freq_or_rare_words(line, words_to_remove))
                    
                    if instruction == "REMOVE_RARE":
                        if "nb_words" in dict_instruction.keys() and "apply" not in dict_instruction.keys():
                            print(f"Most rare words in the column {column}\n{most_common[-int(dict_instruction['nb_words']):]}")
                        if "nb_words" in dict_instruction.keys() and "apply" in dict_instruction.keys():
                            print(f"Most rare words in the column {column}\n{most_common[-int(dict_instruction['nb_words']):]}")
                            words_to_remove = [w for (w, word_count) in most_common[-int(dict_instruction["nb_words"]):]]
                            df_copy[column] = df_copy[column].apply(lambda line: remove_freq_or_rare_words(line, words_to_remove))
                    continue
            # stemming
            elif instruction == "STEM":
                if "language" not in dict_instruction.keys():
                    df_copy[columns_to_process] = df_copy[columns_to_process].applymap(lambda line: stem_words(line, "english"))
                else:
                    df_copy[columns_to_process] = df_copy[columns_to_process].applymap(lambda line: stem_words(line, dict_instruction["language"]))
                continue
            # extract regex pattern
            elif instruction == "EXTRACT_REGEX_PATTERN":
                secondary_regex_pattern = None
                new_column_name = columns_to_process
                result_type = "list"
                if "secondary_regex_pattern" in dict_instruction.keys():
                    secondary_regex_pattern = dict_instruction["secondary_regex_pattern"]
                if "new_column_name" in dict_instruction.keys():
                    new_column_name = dict_instruction["new_column_name"]
                if "result_type" in dict_instruction.keys():
                    result_type = dict_instruction["result_type"]
                df_copy[new_column_name] = df_copy[columns_to_process].applymap(lambda line: extract_regex_pattern(line, dict_instruction["regex_pattern"], secondary_regex_pattern, result_type))
                continue
            # functions without additional argument
            elif instruction in instructions_without_argument.keys():
                df_copy[columns_to_process] = df_copy[columns_to_process].applymap(instructions_without_argument[instruction])
                continue
            # instruction not supported
            else:
                pass

        if target == "ALL":
            print(f"All instructions have been applied to all columns.")
        elif target.startswith("COLUMNS"):
            print(f"All instructions have been applied to the columns {list_columns}.")
        else:
            print(f"All instructions have been applied to the column {target}.")
        
    print("All instructions have been applied.")

    return df_copy        