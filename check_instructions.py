import sys
import os
import json
from collections import OrderedDict

"""
This function reads instructions from a json file, validates them against a dictionary
of valid instructions, and reports any errors or mismatches. It checks for the following:
- Validity of the target column(s).
- Validity of the instruction itself.
- Presence of mandatory arguments.
- Validity of the type of each argument.

Parameters:
- dict (dict): A dictionary containing valid instructions and their constraints.
- instructions_file (str): The path to a text file containing instructions.
- df (pandas dataframe): The dataframe to which the instructions will be applied.

Returns:
- None: The function reports errors or mismatches to the console and exits the program if there is an error.

List organization for each type of argument:
- nothing: [""]
- int: ["int"]
- float: ["float"]
- str: ["str"]
- list: ["list"]
- str_inlist: ["str_inlist", list]
- int_inlist: ["int_inlist", list]
- dir: ["dir"]
- file: ["file"]
- dir+: ["dir+"]
- file+: ["file+"]

Example Usage:
check_instructions(dict_check_instructions, "Projet/instructions.txt")

Example Instruction File:
{"ALL":{
    LOWERCASE: {}
    REMOVE_PUNCT: {"punctuation": "!?,"}
    REMOVE_STOPWORDS: {"language": "french"}
    REMOVE_FREQUENT: {"nb_words": 10, "apply": "-e"}
    REMOVE_RARE: {"nb_words": 10, "apply": "-e"}
    STEM: {"language": "english"}
    LEMMATIZE_ENGLISH: {}
    REMOVE_EMOJI: {}
    REMOVE_EMOTICONS: {}
    CONVERT_EMOJIS: {}
    CONVERT_EMOTICONS: {}
    REMOVE_URLS: {}
    REMOVE_HTML: {}
    CHAT_WORDS_CONVERSION: {}
    SPELL_CORRECTION: {}
}

Example Dictionary (dict_check_instructions):
dict_check_instructions = {"LOWERCASE": {},
                           "REMOVE_PUNCT": {"punctuation": [""]},
                           "REMOVE_STOPWORDS": {"language": ["liste_str", stopwords.fileids()]},
                            "REMOVE_FREQUENT": {"nb_words": ["int", "mandatory"], "apply": ["liste_str", ["-e"]]},
                            "REMOVE_RARE": {"nb_words": ["int", "mandatory"], "apply": ["liste_str", ["-e"]]},
                            "STEM": {"language": ["liste_str", SnowballStemmer.languages]},
                            "LEMMATIZE_ENGLISH": {},
                            "REMOVE_EMOJI": {},
                            "REMOVE_EMOTICONS": {},
                            "CONVERT_EMOJIS": {},
                            "CONVERT_EMOTICONS": {},
                            "REMOVE_URLS": {},
                            "REMOVE_HTML": {},
                            "CHAT_WORDS_CONVERSION": {},
                            "SPELL_CORRECTION": {}}

Put mandatory at the end of the list of an argument if it is mandatory.
"""


def check_instructions(dict, instructions_file, df):
    with open(instructions_file, 'r') as json_file:
        json_dict = json.load(json_file, object_pairs_hook=OrderedDict)

    # text that will contain all the errors
    text = ""
    # browse through the targets
    for target, value in json_dict.items():
        dict_target = json_dict[target]
        if target.startswith("COLUMN"):
            list_columns = target.split(";")[1:]
            for column in list_columns:
                if column not in df.columns:
                    text += f"Error: {column} : unknown column.\n"
        elif target not in df.columns and target != "ALL":
            text += f"Error: {target} : unknown column.\n"
        else:
            pass
        # browse through the lists of instructions of the column
        for key, value in dict_target.items():
            instruction = key
            # if the instruction is not in the dictionary, we add an error
            if instruction not in dict.keys():
                # check if it's an instruction with a number (ex: EXTRACT_REGEX_PATTERN_1) if we have to apply the same instruction several times 
                if instruction[:-2] not in dict.keys():
                    text += f"Error: {instruction} : unknown instruction.\n"
                    continue
                else:
                    instruction = instruction[:-2]
            # check if the mandatory arguments are present
            for key, value in dict[instruction].items():
                if dict[instruction][key][-1] == "mandatory" and key not in dict_target[instruction].keys():
                    text += f"Error: {instruction} : argument {key} is mandatory.\n"
            # We check the type of each argument
            for key, value in dict_target[instruction].items():
                # check if the argument is a valid argument
                if key not in dict[instruction].keys():
                    text += f"Error: {instruction} : argument {key} is not a valid argument.\n"
                    continue
                # nothing
                if dict[instruction][key][0] == "":
                    continue
                # int
                if dict[instruction][key][0] == "int":
                    try:
                        int(dict_target[instruction][key])
                    except ValueError:
                        text += f"Error: {instruction} : argument {key} must be an integer.\n"
                        continue
                # float
                if dict[instruction][key][0] == "float":
                    try:
                        float(dict_target[instruction][key])
                    except ValueError:
                        text += f"Error: {instruction} : argument {key} must be a float.\n"
                        continue
                # str
                if dict[instruction][key][0] == "str":
                    try:
                        str(dict_target[instruction][key])
                    except ValueError:
                        text += f"Error: {instruction} : argument {key} must be a string.\n"
                        continue
                # list
                if dict[instruction][key][0] == "list":
                    if isinstance(dict_target[instruction][key], list) == False:
                        text += f"Error: {instruction} : argument {key} must be a list.\n"
                        continue
                # str_inlist
                if dict[instruction][key][0] == "str_inlist":
                    if dict_target[instruction][key] not in dict[instruction][key][1]:
                        text += f"Error: {instruction} : argument {key} must be in {dict[instruction][key][1]}.\n"
                        continue
                # int_inlist
                if dict[instruction][key][0] == "int_inlist":
                    try:
                        int(dict_target[instruction][key])
                    except ValueError:
                        text += f"Error: {instruction} : argument {key} must be an integer.\n"
                        continue
                    if int(dict_target[instruction][key]) not in dict[instruction][key][1]:
                        text += f"Error: {instruction} : argument {key} must be in {dict[instruction][key][1]}.\n"
                        continue
                # dir
                if dict[instruction][key][0] == "dir":
                    if not os.path.isdir(dict_target[instruction][key]):
                        text += f"Error: {instruction} : argument {key} must be a valid directory.\n"
                        continue
                # file
                if dict[instruction][key][0] == "file":
                    if not os.path.isfile(dict_target[instruction][key]):
                        text += f"Error: {instruction} : argument {key} must be a valid file.\n"
                        continue
                # dir+
                if dict[instruction][key][0] == "dir+":
                    if not os.path.isdir(dict_target[instruction][key]):
                        try:
                            os.mkdir(dict_target[instruction][key])
                        except:
                            text += f"ERROR argument '{instruction}' :  impossible de créer le dossier '{dict_target[instruction][key]}'\n"
                        continue
                # file+
                if dict[instruction][key][0] == "file+":
                    if not os.path.isfile(dict_target[instruction][key]):
                        try:
                            file = open(dict_target[instruction][key], "w")
                            file.close()
                        except:
                            text += f"ERROR argument '{instruction}' :  impossible de créer le fichier '{dict_target[instruction][key]}'\n"
                        continue
                        
    # print the errors   
    print(text)
    # if there is an error, we exit the program
    if text != "":
        sys.exit(1)

    return 
