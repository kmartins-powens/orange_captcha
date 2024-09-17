import os
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from download_orange_image import GatherOrangeImages
import torchvision.transforms as transforms
import torch
import PIL
from train_model import Net, classes


class ButtonClickProcessor(object):
    def __init__(self, axes, label, file, pred = None):
        self.button = Button(axes, label)
        self.button.on_clicked(self.process)
        self.file = file[9:]
        self.pred = pred

    def process(self, event):
        if self.pred:
            os.rename(
                f'/home/kevin/dev/orange_captcha/unsorted/{self.file}',
                f'/home/kevin/dev/orange_captcha/data/{self.pred}/{self.file}'
            )
        else:
            os.rename(
                f'/home/kevin/dev/orange_captcha/unsorted/{self.file}',
                f'/home/kevin/dev/orange_captcha/data/{self.button.label._text}/{self.file}'
            )
        plt.close()

net = Net()
net.load_state_dict(torch.load('/home/kevin/dev/woob/modules/orange/pages/image_classifier.pth', weights_only=True))
net.eval()
transform = transforms.Compose([transforms.PILToTensor()])

for i in range(100):
    files = [f'unsorted/{x}' for x in os.listdir("./unsorted")]
    if not files:
        gather = GatherOrangeImages()
        gather.process()
        files = [f'unsorted/{x}' for x in os.listdir("./unsorted")]
    total = len(files)
    current = 0
    for file in files:
        image = mpimg.imread(file)
        plt.imshow(image)
        i = 0.05
        y = 0.01
        btns = []
        for subj in ("avion", "café", "chat", "cheval", "fleur", "lion", "oiseau", "poisson", "violon", "bateau", "camion", "chaussure", "chien", "éléphant", "girafe", "moto", "papillon", "théière", "tracteur", "tomate", "canard", "tortue", "pred"):
            if i >= 0.94 and y < 0.50:
                i = 0.05
                y = 0.80
            if i >= 0.94 and y >= 0.80:
                i = 0.05
                y = 0.95
            btn_pose = plt.axes([y, i, 0.1, 0.075])
            if subj == "pred":
                pil_image = PIL.Image.open(file)
                torch_image = transform(pil_image).type(torch.float)
                torch_image = torch_image.unsqueeze(0)
                output = net(torch_image)
                pred = torch.max(output.data, 1)
                label = classes[int(pred.indices)]
                btns.append(ButtonClickProcessor(btn_pose, "pred", file, pred=label))
            else:
                btns.append(ButtonClickProcessor(btn_pose, subj, file))
            i += 0.10
        plt.title(f"{label} | {total - current} left", loc="right")
        plt.show()
        current += 1
