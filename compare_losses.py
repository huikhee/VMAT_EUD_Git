import torch
import matplotlib.pyplot as plt

checkpoint = torch.load('Cross_CP/Cross_VMAT_Artifical_data_1500_01Dec_amp_parallel_coll0_checkpoint.pth')


epoch = checkpoint['epoch'] + 1  # Increment the epoch by 1
train_losses_1 = checkpoint['train_losses']
val_losses_1 = checkpoint['val_losses']


checkpoint = torch.load('Cross_CP/Cross_VMAT_Artifical_data_1500_01Dec_amp_parallel_coll0_ResBlock_checkpoint.pth')


epoch = checkpoint['epoch'] + 1  # Increment the epoch by 1
train_losses_2 = checkpoint['train_losses']
val_losses_2 = checkpoint['val_losses']

# plot the train and val loss

plt.plot(train_losses_1, label='train_losses_1')
plt.plot(val_losses_1, label='val_losses_1')
plt.plot(train_losses_2, label='train_losses_2')
plt.plot(val_losses_2, label='val_losses_2')
plt.yscale('log')
plt.legend()
plt.show()