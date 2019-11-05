# Data Processing

We downloaded the raw NHANES survey data (using the Python package [NHANES-Downloader](https://github.com/mrwyattii/NHANES-Downloader). This yielded raw (.json and .xpt) data, along with .csv data for nine surveys: 
* 1999 - 2000
* 2001 - 2002
* 2003 - 2004
* 2005 - 2006
* 2007 - 2008
* 2009 - 2010
* 2011 - 2012
* 2013 - 2014
* 2015 - 2016

And for each of these years, the data were also subdivided into the survey's five component parts: 
* Demographics
* Dietary
* Examination
* Laboratory
* Questionnaire 

The raw data files are all contained in the `csv_data` folder. 

To pull variable descriptions from the variable codes, we downloaded the variable lists for each of the five components, available on the CDC website delineated by component: [Demographics](https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component=Demographics), [Dietary](https://wwwn.cdc.gov/nchs/nhanes/Search/variablelist.aspx?Component=Dietary), [Examination](https://wwwn.cdc.gov/nchs/nhanes/Search/variablelist.aspx?Component=Examination), [Laboratory](https://wwwn.cdc.gov/nchs/nhanes/Search/variablelist.aspx?Component=Laboratory), [Questionnaire](https://wwwn.cdc.gov/nchs/nhanes/Search/variablelist.aspx?Component=Questionnaire). `NHANES-varnames_raw.xlsx` contains the consolidated variable lists. 

This yielded 13,407 distinct variables. From this, we manually subsetted the set of features to include in our analysis to just those variables that were relevant in the context of food security and related health outcomes. The file `NHANES-varnames_yesflag.csv` includes the manually generated binary flag indicating the variables we included. 

We then looped over the full set of survey years and components to create a cleaned file where each row represents a unique individual (identified by the `SEQN` identifier across tables) and survey year combination, and only the subsetted columns are included. `NHANES-clean.csv` contains the final version of our dataset. 

The script to run this data cleaning is contained in `process_data.py`. 