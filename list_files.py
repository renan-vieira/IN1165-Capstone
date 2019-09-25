import glob
import sys

data_set_option = sys.argv[1]

wav_files = glob.glob(f'{data_set_option}/**/*.WAV', recursive=True)
wav_files = [wf.split('TIMIT/')[-1] + '\n' for wf in wav_files]

data_list_file_name = sys.argv[2]
with open(data_list_file_name, 'w+') as f:
    f.writelines(wav_files)
