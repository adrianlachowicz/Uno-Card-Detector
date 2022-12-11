import torch
import torch.optim as optim
import neptune.new as neptune
from tqdm import tqdm
from model import get_model
from config import (
    DEVICE,
    NUM_EPOCHS,
    OPTIMIZER,
    LR,
    BATCH_SIZE,
    NUM_WORKERS,
    NEPTUNE_API_TOKEN,
)
from argparse import ArgumentParser
from datasets import get_train_dataloader, get_val_dataloader


def parse_args():
    parser = ArgumentParser(
        description="The script trains the model with parameters specified in a config file."
    )
    args = parser.parse_args()
    return args


def train_batch(model, data, optimizer):
    """
    The function trains model using passed data batch.

    Arguments:
        model (FasterRCNN) - The FasterRCNN model to train.
        data (list) - The data, which will be used for training model in shape [inputs, targets]
        optimizer (torch.optim.*) - The optimizer to optimize model parameters.

    Returns:
        loss (torch.tensor) - Tensor, which contains sum of specific losses returned from model.
        losses (dict) - Dictionary with values of specific losses.
    """

    model.train()
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

    return loss, losses


@torch.no_grad()
def validation_batch(model, data):
    """
    The function passes validation data across the model and returns losses calculated on data.
    The model is in train mode because on the evaluation mode it returns bounding boxes etc. but
    we want a loss calculated on passed data.

    Arguments:
        model (FasterRCNN) - The FasterRCNN model to train.
        data (list) - The data, which will be used for training model in shape [inputs, targets]

    Returns:
        loss (torch.tensor) - Tensor, which contains sum of specific losses returned from model.
        losses (dict) - Dictionary with values of specific losses.
    """

    model.train()
    inputs, targets = data

    # Move tensors to specific device
    inputs = [input_.to(DEVICE) for input_ in inputs]

    for target in targets:
        for k, v in target.items():
            target[k] = v.to(DEVICE)

    losses = model(inputs, targets)
    loss = sum([loss_ for loss_ in losses.values()])

    return loss, losses


def train_model():
    """
    The function trains FasterRCNN model and logs training data to Neptune AI service.
    """

    # Define dataloaders
    train_dataloader = get_train_dataloader()
    val_dataloader = get_val_dataloader()

    # Get model
    model = get_model()
    model.to(DEVICE)

    # Define optimizer
    optimizer = getattr(optim, OPTIMIZER)(model.parameters(), lr=LR)

    run = neptune.init_run(
        project="adrianlachowicz/UnoDetector",
        api_token=NEPTUNE_API_TOKEN,
    )

    params = {
        "optimizer": OPTIMIZER,
        "learning_rate": LR,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
    }

    run["parameters"] = params

    # Train model
    for epoch in range(1, NUM_EPOCHS + 1):

        print("Epoch {}: starting... \n".format(epoch))

        print("Training started...")
        train_running_loss = 0.0

        train_p_bar = tqdm(total=len(train_dataloader), position=0, leave=True)

        for i, data in enumerate(train_dataloader):
            loss, losses = train_batch(model, data, optimizer)

            # Log step loss
            run["train/step_loss"].log(loss)

            train_running_loss += loss
            train_p_bar.update(1)

        # Log validation epoch loss
        train_average_loss = train_running_loss / len(train_dataloader)
        run["train/epoch_loss"].log(train_average_loss)

        print("\nValidation started...")
        val_running_loss = 0.0

        val_p_bar = tqdm(total=len(val_dataloader), position=0, leave=True)

        for i, data in enumerate(val_dataloader):
            loss, losses = validation_batch(model, data)

            # Log validation step loss
            run["validation/step_loss"].log(loss)

            val_running_loss += loss
            val_p_bar.update(1)

        # Log validation epoch loss
        validation_average_loss = val_running_loss / len(val_dataloader)
        run["validation/epoch_loss"].log(validation_average_loss)

        print("Saving model to file...")
        torch.save(model.state_dict(), "./outputs/model-{}.pth".format(epoch))

        torch.cuda.empty_cache()

        print("\nEpoch {}: end... \n".format(epoch))

    run.stop()


if __name__ == "__main__":
    train_model()
