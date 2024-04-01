import pandas as pd
from pathlib import Path
import re
import itertools

extra_replacements = {'C=CH': 'CH=C',
                      'C=CH2': 'CH2=C',
                      'O': '-O-',
                      'CH-SH': 'CHSH',
                      'CH2-SH': 'CH2SH',
                      'CH-O-': 'CH-O',
                      'CH-S-':'CHS',
                      'CH2-S-':'CH2S',
                      'CH=CH2':'CH2=CH',
                      '>Ncyc': 'N (cyclic)',
                      'Ccyc': 'C (cyclic)',
                      '(N=C)cyc': 'C=N (cyclic)',
                      '(CH2=C)cyc': 'CH2=C (cyclic)',
                      '(CH=C)cyc': 'CH=C (cyclic)',
                      '(C=C)cyc': 'C=C (cyclic)',
                      '(CH=CH)cyc': 'CH=CH (cyclic)',
                      'CH-CN': 'CHCN',
                      'CH-CO-': 'CHCO',
                      'N-CH3': 'CH3N',
                      'N-CH2': 'CH2N',
                      'CH2-NO2': 'CH2NO2',
                      'CH-NO2': 'CHNO2',
                      'COO': 'COO except as above',
                      'CH2-O-': 'CH2O',
                      'CH2-CN': 'CH2CN',
                      'CH-CN': 'CHCN',
                      'CH2-CO-': 'CH2CO',
                      'N-CH': 'CH-N'
                      }

def filter_regex(df, regex):
        return df[df['Group'].str.contains(regex)].iloc[:,:2]

        
def load_data():
    df = pd.read_excel('data/second_order.xlsx', sheet_name='Sheet1')
    df.index = df['No']

    return df

def filter_sidechains(df):
    has_sidechains = filter_regex(df, r'[()]')
    has_sidechains = has_sidechains[~has_sidechains['Group'].str.contains(r'[-.()=]*cyc[-.()=\[\]]*')]
    return has_sidechains

def extract_sidechains(df):
    pass

def filter_alternations(df):
    return filter_regex(df, r'aC')

def extract_alternations(df):
    return df['Group'].str.extract(r'(^[-\w\s|]*)\sor\s([-\w\s|]*)').dropna()

def filter_var_mult(df):
    return filter_regex(df, r'[mnpk]')

def extract_var_mult(df):
    return df['Group'].str.extract(r'(?P<query>\S*)\s(?P<range_str>\[.*\])*').dropna()

def filter_aroring(df):
    return filter_regex(df, r'(AROMRING|PYRIDINE)')

def extract_aroring(df):
    dfr = filter_aroring(df)
    dfr['type'] = dfr['Group'].str.extract('(AROMRING|PYRIDINE)')[0]
    dfr['positions'] = dfr['Group'].str.extractall('s(\d)').groupby(level=0)[0].apply(list)
    dfr = dfr.drop(columns=['No', 'Group'])
    return dfr

def filter_cyclic(df):
    return filter_regex(df, r'cyc')

def extract_cyclic(df):
    parens = filter_regex(df, r'cyc')
    regex1 = r'^(\(?[^()]*\)?)\(?(?:\(?(cyc)\)?)[-|](.*)$'
    r1 = parens['Group'].str.extract(regex1).dropna()
    r1[0] = r1[0] + r1[1]
    r1[1] = r1[2]
    return r1.drop(columns=2)
    

def filter_halogen(df):
    return filter_regex(df, r'X')

def extract_halogen(df):
    dfr = filter_regex(df, r'X')
    pass

def other(df):
    sidechains_idx = set(filter_sidechains(df).index.values.tolist())
    alternations_idx = set(filter_alternations(df).index.values.tolist())
    var_mult_idx = set(filter_var_mult(df).index.values.tolist())
    aroring_idx = set(filter_aroring(df).index.values.tolist())
    cyclic_idx = set(filter_cyclic(df).index.values.tolist())
    all_idx = list(sidechains_idx.union(alternations_idx, var_mult_idx, aroring_idx, cyclic_idx))
    return df.drop(all_idx)

def map_ranges(df):
  range_str = df['range_str']
  df_sub = range_str.str.extract(r'\[(?P<var>[\W,\d\S]*)\sin\s(?P<range_start>\d)..(?P<range_stop>\d)\]')
  df_sub['var'] = df_sub['var'].str.split(',')
  df_sub['perms'] = pd.Series(df_sub.apply(gen_number_perms, axis=1))
  df = df_sub.insert(0, 'query', df['query'])
  return df_sub

def gen_number_perms(series):
  n = len(series['var'])
  return list(itertools.product(list(range(int(series['range_start']), 
                                    int(series['range_stop']) + 1)), 
                                    repeat=n))

def gen_query_perms_numeric(df, series):
  dataset = df
  first_order_groups = dataset['Group']

  queries = []
  for p in series['perms']:
    i = 0
    q = ""
    for idx, v in series.iloc[:11].iteritems():
        if v is None:
          continue
        if re.search(r'[mnpk]', v):
          if (p[i] == 0):
            q += v[:-2]
          elif (p[i] == 1):
            group_name = re.sub(r'[mnpk]', "", v)
            q += group_name
          else:
            group_name = re.sub(r'[mnpk]', str(p[i]), v)
            q += group_name
          i += 1
        else:
          q += v
    queries.append(q)
  # print(queries)
  return queries



if __name__ == "__main__":
    pass