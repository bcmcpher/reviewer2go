"""
Load the Reproschema to build LLM prompts.
"""

import json
from pathlib import Path
import re


# look for protocols
protocol = "cobidas"

# path to files
root = "/home/bcmcpher/Projects/brainhack2023/reviewer2go/cobidas_schema"
outs = "/home/bcmcpher/Projects/brainhack2023/reviewer2go/bin/cobidas-questions.json"

# paths to the folders
label_folder = Path(root, "labels", protocol)
schema_folder = Path(root, "schemas")
protocol_file = Path(schema_folder, protocol, "protocols", f"{protocol}_schema.jsonld")

# for some reason, the pythonic relative paths wouldn't resolve?

# load the protocol file
with open(protocol_file) as f:
    protocol_json = json.load(f)

# extract the list of individual protocols that are part of this schema
activities_order = protocol_json["ui"]["order"]

# initialize output
output = {}

# for every individual schema
for i, activity in enumerate(activities_order):

    # extract the activity features
    activity_file = protocol_file.parent / activity
    activity_name = activity_file.stem.replace("_schema", "")
    # print(f"Processing activity #{i+1:>01}: {activity_name}")

    # load the activity schema
    with open(activity_file) as f:
        activity_json = json.load(f)

    # pull the activity elements
    items_order = activity_json["ui"]["order"]

    # create an empty list of questions
    questions = []

    # for every question in the activity
    for j, item in enumerate(items_order):

        # load the questions schema
        with open(activity_file.parent / item) as f:
            item_json = json.load(f)

        # pull the english question stem
        question = item_json["question"]["en"]

        # remove anything after <div>
        if "<div" in question:
            question = question.split("<div")[0]

        # clean the question stem
        cquest = re.sub(".* - ", "", question)

        # add the text of the question
        questions.append(cquest)

        # print with new structures to track
        # print(f" -- {i+1:>02}.{j:>02} - {cquest}")

    # store the questions
    output[activity_name] = questions

# save questions to file
with open(outs, 'w') as fp:
    json.dump(output, fp)
