#!/bin/bash

export FLASK_APP=server/app.py

# create uploads directory
mkdir -p server/static/uploads

flask run