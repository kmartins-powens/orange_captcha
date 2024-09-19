import torchvision
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
from matplotlib import pyplot as plt

args = argparse.ArgumentParser()
args.add_argument("-t", action="store_true", help="Train the model")

classes = [
    "avion",
    "bateau",
    "café",
    "camion",
    "canard",
    "chat",
    "chaussure",
    "cheval",
    "chien",
    "fleur",
    "girafe",
    "lion",
    "moto",
    "oiseau",
    "papillon",
    "poisson",
    "théière",
    "tomate",
    "tortue",
    "tracteur",
    "violon",
    "éléphant",
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(3600, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 22)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    options = args.parse_args()

    # INIT DATASET
    transform = v2.Compose(
        [
            v2.ColorJitter(brightness=0.5, hue=0.3),
            v2.RandomAdjustSharpness(sharpness_factor=2),
            v2.RandomHorizontalFlip(),
            # v2.RandomRotation([1, 20]),
            # v2.RandomPerspective(),
            ToTensor(),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root="./data", transform=transform)
    # SPLIT DATASET
    splits = [0.8, 0.1, 0.1]
    split_sizes = []
    for sp in splits[:-1]:
        split_sizes.append(int(sp * len(dataset)))
    split_sizes.append(len(dataset) - sum(split_sizes))
    train_set, test_set, val_set = torch.utils.data.random_split(dataset, split_sizes)

    # DATALOADERS
    dataloaders = {
        "train": DataLoader(train_set, batch_size=64, shuffle=True),
        "test": DataLoader(test_set, batch_size=32, shuffle=False),
        "val": DataLoader(val_set, batch_size=32, shuffle=False),
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = "image_classifier.pth"
    training_loss = []
    validation_loss = []
    if options.t:
        # DEFINE MODEL
        net = Net()
        net.to(device)
        net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

        best_val_loss = float('inf')
        patience = 10
        epochs_no_improve = 0
        # TRAIN MODEL
        for epoch in range(300):  # loop over the dataset multiple times
            running_loss = 0.0
            net.train()
            for i, data in enumerate(dataloaders["train"], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 0:
                    print(f"[{epoch + 1}, {i + 1:2d}] loss: {running_loss / 50:.3f}")
                running_loss = 0.0

            training_loss.append(loss.item()/50)
            # Évaluation sur le jeu de validation
            net.eval()  # Mettre le modèle en mode évaluation
            val_loss = 0.0
            with torch.no_grad():
                for data in dataloaders["val"]:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(dataloaders["val"])  # Moyenne de la perte de validation
            print(f"Epoch {epoch} Validation Loss: {val_loss/50:.3f}")
            validation_loss.append(val_loss/50)
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
            # Appeler le scheduler avec la perte de validation
            scheduler.step(val_loss)

        print(f"Finished Training.")
        torch.save(net.state_dict(), model_path)

    # plot model perfomance
    fig, axs = plt.subplots(2)
    fig.suptitle('training loss against validation loss')
    x = 0
    for y1, y2 in zip(training_loss, validation_loss):
        axs[0].plot(x, y1)
        axs[1].plot(x, y2)
        x += 1
    plt.show()

    dataiter = iter(dataloaders["test"])
    images, labels = next(dataiter)

    print("Test sample")
    print(
        "GroundTruth: ", " ".join(f"{dataset.classes[labels[j]]:10s}" for j in range(4))
    )
    net = Net()
    net.eval()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print(
        "Predicted: ",
        " ".join(f"{dataset.classes[predicted[j]]:10s}" for j in range(4)),
    )

    for d in ("test", "val"):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in dataloaders[d]:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f'Accuracy of the network on the {len(test_set) if d == "test" else len(val_set)} {d} images: {100 * correct // total} %'
        )

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in dataset.classes}
    total_pred = {classname: 0 for classname in dataset.classes}

    # again no gradients needed
    with torch.no_grad():
        for data in dataloaders["val"]:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[dataset.classes[label]] += 1
                total_pred[dataset.classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        try:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        except ZeroDivisionError:
            accuracy = 0
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
