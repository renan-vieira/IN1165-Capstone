import sys

set_option = sys.argv[1]
set_option = set_option.lower() if any(l.isupper() for l in set_option) else set_option

file = f'SincNet_model/data_lists/TIMIT_{set_option}.txt'

with open(file, 'r') as f:
    file_uppercase_names = [name.upper() for name in f]

with open(file, 'w') as f:
    f.writelines(file_uppercase_names)



