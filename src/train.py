import numpy as np
import torch
import torch.nn as nn
import os

from models.ardizzone import Model
from tester import test_inverse, test_model
from utils.dataloader import FKIKDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.mmd import MMD

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = FKIKDataset("fk_ik_dataset.pt")
    train_dataset, val_dataset = random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    mask = torch.zeros(dataset[0]['joint_config'].shape[0])
    mask[:len(mask)//2] = 1
    mask = mask.bool()

    model = Model(nr_blocks=2, masks=[mask, mask]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Loss functions for the 2 components - need to be changed later
    loss_fn_y = nn.MSELoss()
    loss_fn_z = nn.MSELoss()

    nr_epochs = 200
    mmd_scale = 1.0  # Increased weight for MMD loss to enforce standard normal
    gen_type = 'noise'
    noise_scale = 0.01  # Increased noise scale
    
    # Set up checkpoint directory (outside the loop)
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {os.path.abspath(checkpoint_dir)}")

    writer = SummaryWriter(log_dir='runs/experiment_1')

    for epoch in range(nr_epochs):
        model.train()
        total_loss = 0.0
        for train_batch in train_loader:
            x = train_batch['joint_config']
            y = train_batch['ee_pose']

            pred = model(x)
            y_pred = pred[:, :2]
            z_pred = pred[:, 2:]

            # 1. Prediction loss
            loss_y = loss_fn_y(y_pred, y)

            # 2. Reconstruction loss
            if gen_type=='normal':
                z_latent = torch.randn_like(z_pred)  # Fixed: use standard normal
            elif gen_type=='noise':
                z_latent = z_pred + noise_scale*torch.randn_like(z_pred)  # Fixed: use normal noise

            pred_recon = torch.concatenate((y, z_latent), dim=1)
            x_recon = model.inverse(pred_recon)
            loss_recon = loss_fn_y(x_recon, x)
            
            # 3. Distribution loss - enforce z_pred ~ N(0,1)
            z_target = torch.randn_like(z_pred)  # Fixed: use standard normal target
            loss_z = MMD(z_pred, z_target, "rbf")
            
            loss = loss_y + loss_recon + mmd_scale * loss_z

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            
            total_loss += loss.item()  
        
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        print(f"Epoch {epoch+1}/{nr_epochs}, Average Loss: {avg_loss:.6f}")

        if (epoch+1)%5 == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad(): 
                for val_batch in val_loader:
                    x = val_batch['joint_config']
                    y = val_batch['ee_pose']

                    pred = model(x)
                    y_pred = pred[:, :2]
                    z_pred = pred[:, 2:]

                    # 1. Prediction loss
                    loss_y = loss_fn_y(y_pred, y)

                    # 2. Reconstruction loss
                    if gen_type=='normal':
                        z_latent = torch.randn_like(z_pred)  # Fixed: use standard normal
                    elif gen_type=='noise':
                        z_latent = z_pred + noise_scale*torch.randn_like(z_pred)  # Fixed: use normal noise

                    pred_recon = torch.concatenate((y, z_latent), dim=1)
                    x_recon = model.inverse(pred_recon)
                    loss_recon = loss_fn_y(x_recon, x)
                    
                    # 3. Distribution loss - enforce z_pred ~ N(0,1)
                    z_target = torch.randn_like(z_pred)  # Fixed: use standard normal target
                    loss_z = MMD(z_pred, z_target, "rbf")
                    
                    loss = loss_y + loss_recon + mmd_scale * loss_z
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            print(f"Validation Loss: {avg_val_loss:.6f}")
            
            # Save checkpoint every 5 epochs
            # checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pth"
            # torch.save(model.state_dict(), checkpoint_path)
            # print(f"Model saved to {os.path.abspath(checkpoint_path)}")

    # Save final model
    final_model_path = f"{checkpoint_dir}/final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'nr_blocks': model.nr_blocks,
        'masks': model.masks
    }, final_model_path)
    print(f"Final model saved to {os.path.abspath(final_model_path)}")
    
    writer.close()

    return model, train_dataset

if __name__ == "__main__":
    trained_model, train_dataset = train()


    # Testing the invertibility
    test_inverse(trained_model)

    # Testing the test_model function
    test_model(trained_model, train_dataset, 'cpu')


    

    

