
# speaker_id.py
# Mirco Ravanelli
# Mila - University of Montreal

# July 2018

# Description:
# This code performs a speaker_id experiments with SincNet.

# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os
import time
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from data_io import ReadList, read_conf, str_to_bool
import seaborn as sns
sns.set()


def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict,
                       fact_amp, split, current_batch):
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch = np.zeros([batch_size, wlen])
    lab_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size=batch_size)

    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

    missing_audios = []
    for i in range(batch_size):

        file = data_folder + wav_lst[snt_id_arr[i]]

        if os.path.exists(file):
            # select a random sentence from the list
            signal, _ = sf.read(file)

            # accessing to a random chunk
            snt_len = signal.shape[0]
            snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
            snt_end = snt_beg + wlen

            channels = len(signal.shape)
            if channels == 2:
                print('WARNING: stereo to mono: ' + file)
                signal = signal[:, 0]

            sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
            lab_batch[i] = lab_dict[wav_lst[snt_id_arr[i]].lower()]
        else:
            missing_audios.append('\n' + file)
            print(f'file does note exist {file}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inp = Variable(torch.from_numpy(sig_batch).float().to(device).contiguous())
    lab = Variable(torch.from_numpy(lab_batch).float().to(device).contiguous())

    output_file = f'../missing_{split}_audios.txt'

    with open(output_file, 'a+') as f:
        f.write(f'\nBATCH #{current_batch}\n')
        f.writelines(missing_audios)

    return inp, lab


# Reading cfg file
options = read_conf()

# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder
output_folder = options.output_folder

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

# [dnn]
fc_lay = list(map(int, options.fc_lay.split(',')))
fc_drop = list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act = list(map(str, options.fc_act.split(',')))

# [class]
class_lay = list(map(int, options.class_lay.split(',')))
class_drop = list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act = list(map(str, options.class_act.split(',')))

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# training list
wav_lst_tr = ReadList(tr_lst)
snt_tr = len(wav_lst_tr)
# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop,
            }

CNN_net = CNN(CNN_arch)
CNN_net.to(device)

# Loading label dictionary
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
lab_dict = np.load(class_dict_file).item()

np.load = np_load_old

DNN1_arch = {'input_dim': CNN_net.out_dim,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act,
             }

DNN1_net = MLP(DNN1_arch)
DNN1_net.to(device)

DNN2_arch = {'input_dim': fc_lay[-1],
             'fc_lay': class_lay,
             'fc_drop': class_drop,
             'fc_use_batchnorm': class_use_batchnorm,
             'fc_use_laynorm': class_use_laynorm,
             'fc_use_laynorm_inp': class_use_laynorm_inp,
             'fc_use_batchnorm_inp': class_use_batchnorm_inp,
             'fc_act': class_act,
             }

DNN2_net = MLP(DNN2_arch)
DNN2_net.to(device)

if pt_file != 'none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
    DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

train_losses=[]
train_error_rates = []
val_losses =[]
val_error_rates=[]
for epoch in range(N_epochs):

    print(f'EPOCH #{epoch}')
    start_time = time.time()

    test_flag = 0
    CNN_net.train()
    DNN1_net.train()
    DNN2_net.train()

    loss_sum = 0
    err_sum = 0

    #train_losses = []
    #train_error_rates = []
    #val_losses = []
    #val_error_rates = []


    for i in range(N_batches):
        if (i % 100) == 0:
            print(f'\tBATCH #{i}')

        [inp, lab] = create_batches_rnd(batch_size, data_folder, wav_lst_tr,
                                        snt_tr, wlen, lab_dict, 0.2, split='train', current_batch=i)
        pout = DNN2_net(DNN1_net(CNN_net(inp)))

        pred = torch.max(pout, dim=1)[1]
        loss = cost(pout, lab.long())
        err = torch.mean((pred != lab.long()).float())

        optimizer_CNN.zero_grad()
        optimizer_DNN1.zero_grad()
        optimizer_DNN2.zero_grad()

        loss.backward()
        optimizer_CNN.step()
        optimizer_DNN1.step()
        optimizer_DNN2.step()

        loss_sum = loss_sum + loss.detach()
        err_sum = err_sum + err.detach()

    loss_tot = loss_sum / N_batches
    err_tot = err_sum / N_batches

    end_time = time.time()
    print(f'ELAPSED TIME FOR EPOCH #{epoch}: {(end_time - start_time) / 60.0} min')


    #######################################
    train_losses.append(loss_tot.item())
    train_error_rates.append(err_tot.item())
    print(train_losses,train_error_rates)
    ########################################



    # Full Validation  new
    if epoch % N_eval_epoch == 0:

        start_time = time.time()

        CNN_net.eval()
        DNN1_net.eval()
        DNN2_net.eval()
        test_flag = 1
        loss_sum = 0
        err_sum = 0
        err_sum_snt = 0

        with torch.no_grad():
            for i in range(snt_te):

                [signal, fs] = sf.read(data_folder + wav_lst_te[i])

                signal = torch.from_numpy(signal).float().to(device).contiguous()
                lab_batch = lab_dict[wav_lst_te[i].lower()]

                # split signals into chunks
                beg_samp = 0
                end_samp = wlen

                N_fr = int((signal.shape[0] - wlen) / (wshift))

                sig_arr = torch.zeros([Batch_dev, wlen]).float().to(device).contiguous()
                lab = Variable((torch.zeros(N_fr + 1) + lab_batch).to(device).contiguous().long())
                pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().to(device).contiguous())
                count_fr = 0
                count_fr_tot = 0
                while end_samp < signal.shape[0]:
                    sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                    beg_samp = beg_samp + wshift
                    end_samp = beg_samp + wlen
                    count_fr = count_fr + 1
                    count_fr_tot = count_fr_tot + 1
                    if count_fr == Batch_dev:
                        inp = Variable(sig_arr)
                        pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
                        count_fr = 0
                        sig_arr = torch.zeros([Batch_dev, wlen]).float().to(device).contiguous()

                if count_fr > 0:
                    inp = Variable(sig_arr[0:count_fr])
                    pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))

                pred = torch.max(pout, dim=1)[1]
                loss = cost(pout, lab.long())
                err = torch.mean((pred != lab.long()).float())

                [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                err_sum_snt = err_sum_snt + (best_class != lab[0]).float()

                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()

            err_tot_dev_snt = err_sum_snt / snt_te
            loss_tot_dev = loss_sum / snt_te
            err_tot_dev = err_sum / snt_te

        print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
            epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        ################################################################################
        val_losses.append(loss_tot_dev.item())
        val_error_rates.append(err_tot_dev.item())
        print(val_losses,val_error_rates)
        ################################################################################

        with open(output_folder + "/res.res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                      'DNN1_model_par': DNN1_net.state_dict(),
                      'DNN2_model_par': DNN2_net.state_dict(),
                      }
        torch.save(checkpoint, output_folder + '/model_raw.pkl')
        torch.save(checkpoint, output_folder + '/model_raw.onnx')

        end_time = time.time()
        print(f'EVALUATION ELAPSED TIME: {(start_time - end_time) / 60.0}')

    else:
        print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))

# Loss line
plt.figure(0)
plt.plot(train_losses,color='blue',linewidth=1)
plt.plot(val_losses, color='red', linewidth=1)
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.legend(['training set', 'validation set'])
plt.savefig('losses.png')

# Error line
plt.figure(1)
plt.plot(train_error_rates, color = 'red', linewidth=1)
plt.plot(val_error_rates, color='blue', linewidth=1)
plt.xlabel("# Epochs")
plt.ylabel("Error")
plt.legend(['training set', 'validation set'])
plt.savefig('error_rates.png')
