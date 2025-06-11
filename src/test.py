from torch.utils.data import DataLoader
from utils.dataloader import FKIKDataset
import torch.nn as nn
import torch.optim as optim
import torch

# FK = joint_config → ee_pose
# IK = ee_pose → joint_config
train_direction = 'fk'  # or 'ik'

dataset = FKIKDataset('fk_ik_dataset.pt')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

input_key  = 'joint_config' if train_direction == 'fk' else 'ee_pose'
output_key = 'ee_pose'      if train_direction == 'fk' else 'joint_config'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = nn.Sequential(
    nn.Linear(dataset[0][input_key].shape[0], 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, dataset[0][output_key].shape[0])
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

n_epochs = 10

for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        x = batch[input_key].to(device)
        y = batch[output_key].to(device)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
