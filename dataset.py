import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import utils


def clean_dataset(dataset_path):
    dataset = np.load(dataset_path)
    for data in dataset:
        onset_data = data[:, :]


def calculate_low_high(dataset_path):
    """14, 115"""
    dataset = np.load(dataset_path)
    print(dataset[:, :, 79])
    l = 127
    h = 0
    num_notes = 0
    for data in dataset:

        onset_data = data[:, :] == 2
        # print(onset_data.shape)
        sustain_data = data[:, :] == 1
        silence_data = data[:, :] == 0
        pr = np.stack([onset_data, sustain_data, silence_data],
                      axis=2).astype(bool)
        # pr: (32, 128, 3)
        pr_matrix = utils.piano_roll_to_target(pr)
        # piano_grid = utils.target_to_3dtarget(pr_matrix, max_note_count=20,
        #                                       max_pitch=127,
        #                                       min_pitch=0, pitch_pad_ind=110,
        #                                       pitch_sos_ind=128,
        #                                       pitch_eos_ind=129)
        # try:
        low, high, _, nn = utils.get_low_high_dur_count(pr_matrix)
        nn = nn.max()
        if low < l:
            l = low
        if high > h:
            h = high
        if nn > num_notes:
            num_notes = nn

            # print(low, high)
        # except ValueError:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     ax = plt.subplot(131)
        #     ax.imshow(pr[:, :, 0])
        #     ax = plt.subplot(132)
        #     ax.imshow(pr[:, :, 1])
        #     ax = plt.subplot(133)
        #     ax.imshow(pr[:, :, 2])
        #     plt.show()
    return l, h, num_notes


class PolyphonicDataset(Dataset):

    def __init__(self, filepath, shift_low, shift_high):
        super(PolyphonicDataset, self).__init__()
        self.filepath = filepath
        self.shift_low = shift_low
        self.shift_high = shift_high
        self.data = np.load(self.filepath)

    def __len__(self):
        # consider data augmentation here
        return len(self.data) * (self.shift_high - self.shift_low + 1)

    def __getitem__(self, id):
        # separate id into (no, shift) pair
        no = id // (self.shift_high - self.shift_low + 1)
        shift = id % (self.shift_high - self.shift_low + 1) + self.shift_low
        data = self.data[no, :, :]
        # perform pitch shifting using np.roll and masking
        data = np.roll(data, shift, axis=1)
        if shift > 0:
            data[:, :shift] = 0
        elif shift < 0:
            data[:, shift:] = 0
        # if you want to convert data into MIDI message,
        # insert your converter code after this line.
        onset_data = data[:, :] == 2
        sustain_data = data[:, :] == 1
        silence_data = data[:, :] == 0
        pr = np.stack([onset_data, sustain_data, silence_data],
                      axis=2).astype(bool)
        # pr: (32, 128, 3)
        pr_matrix = utils.piano_roll_to_target(pr)
        # 14 - 115, 14
        # 21 - 105， 11
        piano_grid = utils.target_to_3dtarget(pr_matrix, max_note_count=16,
                                              max_pitch=128,
                                              min_pitch=0, pitch_pad_ind=130,
                                              pitch_sos_ind=128,
                                              pitch_eos_ind=129)
        return piano_grid.astype(np.int64)


def reshape_grid_to_plot(piano_grid):
    y = np.copy(piano_grid)
    y = piano_grid.transpose([1, 0, 2]).reshape((-1, 6 * 32))
    return y


if __name__ == '__main__':
    file_path = \
        r'D:\working\seq2seq-AccArrangement\poly-vae\dataset\pop909+mlpv_t32_fix1'
    # l, h, num_notes = calculate_low_high(file_path + '/pop909+mlpv_t32_val_fix1.npy')
    # print(l, h, num_notes)
    train_dataset = PolyphonicDataset(file_path + '/pop909+mlpv_t32_train_fix1.npy',
                                      -3,
                                      +3)  # DO augment on training dataset!
    val_dataset = PolyphonicDataset(file_path + '/pop909+mlpv_t32_val_fix1.npy', 0, 0)  # DO NOT augment on validation dataset!
    print(len(train_dataset), len(val_dataset))
    print(train_dataset[1124].shape)
    x = train_dataset[1124]
    y = reshape_grid_to_plot(x)
    import matplotlib.pyplot as plt
    plt.imshow(y)
    plt.show()
    # for i in range(0, )

    # # print(train_dataset[(2, 3, 4, 5)])
    # data_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    # for i_batch, sample_batched in enumerate(data_loader):
    #     print(i_batch)
    #     print(len(sample_batched))
    #     print(type(sample_batched))
    #     print(sample_batched.shape)
    #     print(sample_batched[0].shape)
