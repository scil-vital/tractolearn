# -*- coding: utf-8 -*-

"""Labels issues based on title. Must be run in a GitHub Action with the
`issues` event. Does not remove any label the user might have added them
manually.
Adapted from scikit-learn."""

import os
import re

import yaml
from ghapi.all import GhApi, context_github, github_token, user_repo

owner, repo = user_repo()

# Retrieve the issue data
issue = context_github.event.issue
title = issue.title

# Read the list of title and label pairs
filebasename = "title_to_labels.yml"
file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), filebasename
)

with open(file) as f:
    title_to_labels = yaml.safe_load(f.read())

# Add the word boundary to build the title regex
word_boundary = r"\b"
title_to_labels_regex = [
    (word_boundary + key + word_boundary, val)
    for key, val in title_to_labels.items()
]

# Find the label matches
title_labels_to_add = [
    label for regex, label in title_to_labels_regex if re.search(regex, title)
]

# Add labels if matches were found
if title_labels_to_add:

    api = GhApi(owner=owner, repo=repo, token=github_token())
    api.issues.add_labels(issue.number, labels=title_labels_to_add)
