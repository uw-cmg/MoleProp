# Data collection and Parsing tools for Molecular Properties Project.
Data notebooks for the molecular properties project.

## Contributors
    * Nathaniel J. Krakauer
    * James Wang

## Overview
To source and verify correctness, We have 3 jupyter notebooks.
Scrap_Gelest deals with extracting flash points from the Gelest Chemical Catalogue.

Scrap_pubchem extracts flash point values from the PubChem Chemical Database given a json file of ids that have a flash point value.

Integrated Dataset notebook, cleans, parses, and verifies data from academic papers and adds data from Gelest and Pubchem.

Util class provides helper methods to aid in the taskes listed above. PubchemScrapper provides methods used to parse and clean PubChem flash point data.

## Data File
    * cid_flashpoint_list.txt: file containing all chemical IDs of compounds that record a flash point value.

## External Dependencies
    * Python 3.6
    * RDKit
    * [OPSIN](https://opsin.ch.cam.ac.uk/)
    * PyPDF2
    * bs4
    * matplotlib, pandas, numpy