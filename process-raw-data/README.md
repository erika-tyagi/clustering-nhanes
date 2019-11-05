# Data Processing Steps: 

1. We downloaded the raw NHANES survey data (using the Python package [NHANES-Downloader](https://github.com/mrwyattii/NHANES-Downloader). This yielded raw (.json and .xpt) data, along with .csv data for nine surveys: 
* 1999 - 2000
* 2001 - 2002
* 2003 - 2004
* 2005 - 2006
* 2007 - 2008
* 2009 - 2010
* 2011 - 2012
* 2013 - 2014
* 2015 - 2016

For each of these years, the data were also subdivided into the survey's five component parts: 
* Demographics
* Dietary
* Examination
* Laboratory
* Questionnaire 

2. As this yielded 13,407 distinct variables, we manually subsetted the set of features to include in our analysis to just those variables that were relevant in the context of food security and related health outcomes. The file `NHANES-varnames_yesflags.csv` includes the binary flag indicating the variables we included. 

3. We then looped over the full set of survey years and components to create a cleaned file where each row represents a unique individual (identified by the `SEQN` identifier across tables) and survey year combination, and only the subsetted columns are included. 

1. write-up 
2. new vars to keep 
3. EDA 
4. clustering 