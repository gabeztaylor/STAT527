#!/bin/sh

jupyter nbconvert Student.ipynb --TemplateExporter.exclude_input=True --to slides --post serve