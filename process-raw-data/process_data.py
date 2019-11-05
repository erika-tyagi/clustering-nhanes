import pandas as pd
import numpy as np
import glob
import functools

RAW_FILE = 'NHANES-varnames_raw.xlsx'
NO_FLAG_FILE = 'NHANES-varnames_noflag.csv'
YES_FLAG_FILE = 'NHANES-varnames_yesflag.csv'
CLEAN_FILE = 'NHANES-clean.csv'
MISSING_FILE = 'NHANES-missing.csv'

COMPONENTS = ['Laboratory', 'Demographics', 'Questionnaire', 'Dietary', 'Examination']
YEARS = list(range(1999, 2016, 2))

################################################################################

# read raw file and append sheets 
no_flag_df = pd.DataFrame()
for c in COMPONENTS: 
    df = pd.read_excel(RAW_FILE, sheet_name = c)
    no_flag_df = no_flag_df.append(df)
    
# create year flags 
def var_in_year(row, year): 
    if (row['Begin Year'] <= year) & (row['EndYear'] >= year): 
        return 1
    return 0 

for y in YEARS: 
    v = 'flag_' + str(y)
    no_flag_df[v] = no_flag_df.apply(var_in_year, year = y, axis = 1)
    
flags = [col for col in no_flag_df if col.startswith('flag_')]
no_flag_df = (no_flag_df
              .groupby(['Variable Name', 'Variable Description', 'Component', 'Data File Description'])[flags]
              .sum(axis = 1)
              .reset_index())

for y in YEARS: 
    v = 'flag_' + str(y)
    no_flag_df[v] = np.where(no_flag_df[v] == 0, 0, 1)

# write to csv
no_flag_df.to_csv(NO_FLAG_FILE, index = False)

################################################################################

# get list of variables to keep 
# yes_flag_df = pd.read_csv(YES_FLAG_FILE)
# keep_vars = (yes_flag_df[yes_flag_df['keep'] == 1]['Variable Name']
#              .unique()
#              .tolist())
# keep_vars.append('SEQN')

keep_vars = ['DR2TSFAT', 'DR2TSUGR', 'DR2TSODI', 'DR2TIRON', 'DR2TFIBE', 'DR2TCALC', 'DBQ700', 
             'CBD071', 'DBD905', 'DBD895', 'FSD200', 'DBQ424', 'DBQ301', 'DBD910', 'SEQN'] 


# loop over all years and components 
full_df_list = []
for y in YEARS:
    year_df_list = []
    for c in COMPONENTS: 
        path = 'csv_data/' + str(y) + '-' + str(y+1) + '/' + c + '/*.csv'
        for f in glob.iglob(path):
            df = pd.read_csv(f)
            if 'SEQN' not in df.columns: 
                continue
            df = df[df.columns[df.columns.isin(keep_vars)]]
            if df.shape[1] > 1: 
                year_df_list.append(df)
                
    # merge (wide) within each year
    year_df = (functools.reduce(
        lambda df1, df2: pd.merge(df1, df2, on = 'SEQN', how = 'outer'), year_df_list)
               .drop_duplicates(subset = 'SEQN'))
    year_df['year'] = str(y) + '-' + str(y+1)
    full_df_list.append(year_df)

# append (long) across years 
clean_df = pd.concat(full_df_list, axis = 0, sort = True)

# write to csv
clean_df = clean_df[ ['year', 'SEQN'] + [ col for col in clean_df.columns if col not in ['year', 'SEQN'] ] ]
clean_df.to_csv(CLEAN_FILE, index = False)

