import torch.nn as nn
import torch as t
from utils import get_frames,process_frames, SpeedDataset, split_data, get_targets
from model import get_nvidia_model


def train(model, training_generator, device):
    print("Start training")
    return
    epochs = 1
    criterion = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(training_generator, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('finished training')
    return

if __name__ == '__main__':
    train_path = '/Users/pallekc/Jobs/comma/speed_challenge_2017/data/train.mp4'
    test_path = '/Users/pallekc/Jobs/comma/speed_challenge_2017/data/test.mp4'
    train_targets = '/Users/pallekc/Jobs/comma/speed_challenge_2017/data/train.txt'

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    train_frames, train_frames_count = get_frames(train_path)
    train_targets, train_targets_count = get_targets(train_targets)
    print(train_frames_count)
    print(train_targets_count)
    assert train_frames_count == train_targets_count, 'Number of train frames != targets'

    train_processed = process_frames(train_frames, train_targets)  # remember the first one is missing
    train_x, train_y, test_x, test_y = split_data(train_processed)

    train_dataset = SpeedDataset(train_x, train_y)
    training_generator = t.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # train_x, train_y, val_x, val_y = split_data(train_x, train_y)
    # val_dataset = SpeedDataset(val_x, val_y)
    # val_generator = t.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

    model = get_nvidia_model()
    train(model, training_generator, device)