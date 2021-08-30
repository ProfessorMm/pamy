# %%
import torch
import torch.nn as nn
import numpy as np
import pickle5 as pickle
import os
import math
from ConvLSTM import ConvLSTM
import matplotlib.pyplot as plt
# %% set the parameters
# which dof to be trained
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pi = math.pi
out_dof_list = [0]
in_dof_list = [0, 1, 2]
q_input = 10
q_output = 1
length_of_in = 2 * q_input + 1
length_of_output = 2 * q_output + 1
width_in = len( in_dof_list )
width_out = 1
mid_position = q_output
channel_in = 2
k = 0.8
batch_size = 1
dropout = 0.1
learning_rate = 1e-4
weight_decay = 0.00
epochs = 80
min_length = 100000
scr = 10000
# %% read the data from files
file_of_data = 'Data/checked_data_constraints'
path = os.path.realpath( os.curdir )

files = os.listdir( path + '/' + file_of_data )

desired_path_list = []
desired_velocity_list = []
desired_acceleration_list = []

ff_list = []
for file in files:
    if not os.path.isdir( file ):

        abs_path = path + '/' + file_of_data + '/' + file
        f = open(abs_path, 'rb')
        t_stamp_u = pickle.load( f ) # time stamp for x-axis
        y_history = pickle.load( f ) 
        ff_history = pickle.load( f )
        angle_initial = pickle.load( f ) 
        disturbance_history = pickle.load( f )  
        P_history = pickle.load( f )
        time_history = pickle.load( f ) 
        f.close()

        if y_history[0].shape[1] < min_length:
            min_length = y_history[0].shape[1]

        desired_path_list.append( y_history[0] * 180 / pi )
        ff_list.append( ff_history[-1] / scr )

number_path = len( desired_path_list )
t_stamp = np.linspace(0, (min_length - 1) / 100, num=min_length, endpoint=True)
# %% cut the inputs and outputs
total_path = np.array([])

for i in range( number_path ):
    desired_path = desired_path_list[i]
    ff_temp = ff_list[i]

    ff_list[i] = ff_temp[out_dof_list, 0:min_length]
    desired_path_list[i] = desired_path[in_dof_list, 0:min_length]

    if i == 0:
        total_path = np.copy(desired_path_list[i] )
    else:
        total_path = np.hstack((total_path, desired_path_list[i]))

norm = np.zeros( len( in_dof_list ) )
for i in range( len( in_dof_list ) ):
    norm[i] = np.linalg.norm(total_path[i, :], ord=2)

for i in range( number_path ):
    desired_path = desired_path_list[i]
    ff_temp = ff_list[i]

    for dof in range( len( in_dof_list ) ):
        desired_path[dof, :] = desired_path[dof, :] / norm[dof]
    desired_path_list[i] = desired_path

    desired_velocity = np.array([])
    desired_acceleration = np.array([])

    for index in range( min_length ):
        if index == 0:
            desired_velocity = np.zeros( len(in_dof_list ) ).reshape(-1, 1)
            desired_acceleration = np.zeros( len(in_dof_list ) ).reshape(-1, 1)
        else:
            desired_velocity = np.hstack((desired_velocity, (desired_path[:, index].reshape(-1, 1) - desired_path[:, index-1].reshape(-1, 1)) / 0.01 ))
            desired_acceleration = np.hstack((desired_acceleration, (desired_velocity[:, index].reshape(-1, 1) - desired_velocity[:, index-1].reshape(-1, 1)) / 0.01 ))

    desired_velocity_list.append( desired_velocity )
    desired_acceleration_list.append( desired_acceleration )

# %% check
# Location = 'upper center'
# for index in range( number_path ):
#     desired_path = desired_path_list[index]
#     desired_velocity = desired_velocity_list[index]
#     desired_acceleration = desired_acceleration_list[index]
#     line = []
#     plt.figure( index )
#     plt.xlabel(r'time t in s')
#     plt.ylabel(r'Normalization')
#     for dof in range( len( in_dof_list ) ):
#         line_temp, = plt.plot(t_stamp, desired_path[dof, :], linewidth=1, label=r'Path, dof {}'.format(dof))
#         line.append( line_temp )
#         line_temp, = plt.plot(t_stamp, desired_velocity[dof, :], linewidth=1, label=r'Velocity, dof {}'.format(dof))
#         line.append( line_temp )
#         line_temp, = plt.plot(t_stamp, desired_acceleration[dof, :], linewidth=1, label=r'Acceleration, dof {}'.format(dof))
#         line.append( line_temp )
#     plt.legend(handles = line, loc=Location, shadow=True)
#     plt.show()
# %% generate the dataset and labelset
dataset = []
labelset = []

train_data = []
train_label = []
val_data = []
val_label = []

for index in range( number_path ):

    # the dimensions should be 4 * n
    desired_path = desired_path_list[index].T
    desired_velocity = desired_velocity_list[index].T
    desired_acceleration = desired_acceleration_list[index].T
    ff = ff_list[index].T

    l = desired_path.shape[0]

    sliced_data = None
    sliced_label = None

    for i in range( l ):

        y_temp = np.array([])
        v_temp = np.array([])
        a_temp = np.array([])
        u_temp = np.array([])
        data_temp = None
        # loop for inputs
        for j in range( i-q_input, i+q_input+1):
            if j < 0:
                y_temp = np.append(y_temp, desired_path[0, :] )
                v_temp = np.append(v_temp, desired_velocity[0, :] )
                a_temp = np.append(a_temp, desired_acceleration[0, :] )
                
            elif j > l-1:
                y_temp = np.append(y_temp, desired_path[-1, :] )
                v_temp = np.append(v_temp, desired_velocity[-1, :] )
                a_temp = np.append(a_temp, desired_acceleration[-1, :] )
                
            else:
                y_temp = np.append(y_temp, desired_path[j, :] )
                v_temp = np.append(v_temp, desired_velocity[j, :] )
                a_temp = np.append(a_temp, desired_acceleration[j, :] )

            # batch_size * time_length * channels * height * width
        y_temp = torch.tensor( y_temp ).view(1, 1, 1, -1, width_in)
        v_temp = torch.tensor( v_temp ).view(1, 1, 1, -1, width_in)
        a_temp = torch.tensor( a_temp ).view(1, 1, 1, -1, width_in)

        if channel_in == 1:
            data_temp = torch.clone( y_temp )
        elif channel_in == 2:
            data_temp = torch.cat( [y_temp, v_temp], dim=2 )
        elif channel_in == 3:
            data_temp = torch.cat( [y_temp, v_temp, a_temp], dim=2 )
        
        if i == 0:
            sliced_data = torch.clone( data_temp )
        else:
            sliced_data = torch.cat( [sliced_data, data_temp], dim=1 )

        # loop for outputs
        for j in range( i-q_output, i+q_output+1):
            if j < 0:
                u_temp = np.append(u_temp, ff[0, :] )
            elif j > l-1:
                u_temp = np.append(u_temp, ff[-1, :] )
            else:
                u_temp = np.append(u_temp, ff[j, :] )

        u_temp = torch.tensor( u_temp ).view(1, 1, 1, -1, width_out)
        if i == 0:
            sliced_label = torch.clone( u_temp )
        else:
            sliced_label = torch.cat([sliced_label, u_temp], dim=1)
    
    labelset.append( sliced_label )
    dataset.append( sliced_data )

number_train = int( k * number_path )
number_val = number_path - number_train

arr = np.arange( number_path )
# if select training data randomly
# np.random.shuffle( arr )

# seperate the data into train data and validation data
for i in range( number_path ):
    if i < number_train:
        train_data.append( dataset[arr[i]] )
        train_label.append( labelset[arr[i]] )
    else:
        val_data_temp = dataset[arr[i]]
        val_label_temp = labelset[arr[i]]

        val_data.append( val_data_temp ) 
        val_label.append( val_label_temp )

train_loader = []

idx = 0

while 1:
    if idx + batch_size < number_train:
        
        batch_x = torch.cat(train_data[idx:idx+batch_size], dim=0)
        batch_y = torch.cat(train_label[idx:idx+batch_size], dim=0)

        train_loader.append( (batch_x, batch_y) )

        idx += batch_size
    else:
        break
# %% build the model
# batch_size * time_length * channels * height * width
# b * t * c * h * w
# build ConvLSTM model, the input data has three channels,
# model has three LSTM layers, number of hidden states of each layer is 64, 64 and 128
hidden_dim = [32, 16]
kernel_size = [(5, 1), (5, 1)]
# channels * length * width
fc_in_dim = hidden_dim[-1] * (length_of_in - 3 * (5 - 1) ) * 1
fc_out_dim = 1 * length_of_output * 1
model = ConvLSTM( input_dim=channel_in, 
                  hidden_dim=hidden_dim,
                  kernel_size=kernel_size,
                  num_layers=2,
                  fc_in_dim=fc_in_dim,
                  fc_out_dim=fc_out_dim,
                  dropout=dropout,
                  batch_first=True,
                  bias=True )

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)

loss_function = nn.MSELoss( size_average=True ) 
# %% use gpu
model.to(device)
loss_function.to(device)
# %% train the model
iter_per_epoch = len(train_loader)

train_loss_history = []
val_loss_history = []

for epoch in range(epochs):

    avg_train_loss = 0.0 # loss summed over epoch and averaged
    model.train()

    for (batch_data, batch_label) in train_loader:
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        layer_output_list, last_state_list = model(batch_data.float())
        loss = loss_function(layer_output_list[-1].squeeze().float(), batch_label.squeeze().float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_train_loss += loss.item()
    
    avg_train_loss /= iter_per_epoch
    train_loss_history.append(avg_train_loss) 
    # print ('\n[Epoch {}/{}] TRAIN loss: {:.3f}'.format(epoch+1, epoches, avg_train_loss))

    avg_eval_loss = 0.0
    model.eval()

    for i in range( len(val_label) ):      
        layer_output_list, last_state_lists = model(val_data[i].float())
        loss = loss_function(layer_output_list[-1].squeeze().float(), val_label[i].squeeze().float())
        avg_eval_loss += loss.item()
 
    avg_eval_loss /= len(val_label)
    val_loss_history.append(avg_eval_loss)
    print ('\n[Epoch {}/{}] TRAIN/VALID loss: {:.6}/{:.6f}'.format(epoch+1, epochs, avg_train_loss, avg_eval_loss))
# %% draw the plots
plt.figure(1)
plt.plot(train_loss_history, '-o', label='Training')
plt.plot(val_loss_history, '-o', label='Evaluation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(ncol=2, loc='upper center')
plt.show()
# %%
Location = 'upper center'
for index in range( number_val ):
    layer_output_list, last_state_lists = model(val_data[index].float())
    output = layer_output_list[-1].squeeze()
    output = output.detach().numpy()
    output_label = val_label[index].squeeze().detach().numpy()
    line = []

    plt.figure( index )
    plt.xlabel(r'time t in s')
    plt.ylabel(r'Prediction')

    line_temp, = plt.plot(t_stamp, output[:, mid_position], linewidth=1, label=r'Prediction')
    line.append( line_temp )
    line_temp, = plt.plot(t_stamp, output_label[:, mid_position], linewidth=1.5, linestyle='--', label=r'Desired')
    line.append( line_temp )

    plt.legend(handles = line, loc=Location, shadow=True)
    plt.show()


# Location = 'upper center'

# index = 0
# for dof in out_dof_list:

#     line = []
#     plt.figure( r'Prediction for dof: {}'.format(dof) )
#     plt.xlabel(r'time t in s')
#     plt.ylabel(r'Feedforward control')
#     line_temp, = plt.plot(t_stamp, u_model[index, :] * (max_ff[dof] - min_ff[dof]), linewidth=1, label=r'Prediction, dof {}'.format(dof))
#     line.append( line_temp )
#     line_temp, = plt.plot(t_stamp, u_eval[index, :] * (max_ff[dof] - min_ff[dof]), linewidth=1.5, linestyle='--', label=r'Desired, dof {}'.format(dof))
#     line.append( line_temp )
#     plt.legend(handles = line, loc=Location, shadow=True)
#     plt.show()

#     index += 1