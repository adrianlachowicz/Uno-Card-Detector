import torch.optim as optim
from tqdm import tqdm
from model import get_model
from config import DEVICE, NUM_EPOCHS, LR
from argparse import ArgumentParser
from datasets import get_train_dataloader, get_val_dataloader

def parse_args():
    parser = ArgumentParser(description="The script trains the model with parameters specified in a config file.")
    args = parser.parse_args()
    return args

def train_model():
    print("Training on: {}".format(DEVICE))
    
    train_dataloader = get_train_dataloader()
    val_dataloader = get_val_dataloader()

    model = get_model()
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        p_bar = tqdm(total=len(train_dataloader))

        for i, data in enumerate(train_dataloader):
            inputs, targets = data

            # Move tensors to specific device
            inputs = [input_.to(DEVICE) for input_ in inputs]
            
            for target in targets:
                for k, v in target.items():
                    target[k] = v.to(DEVICE)
            
            losses = model(inputs, targets)
            loss = sum([loss_ for loss_ in losses.values()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            p_bar.update(1)
            break

        average_loss = running_loss / len(train_dataloader)
        break
    

if __name__ == "__main__":
    parse_args()
    train_model()