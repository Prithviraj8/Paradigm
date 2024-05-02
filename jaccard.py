import torchmetrics
import torch
import numpy as np
from PIL import Image
import sys
jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

def validate(val_path, predTensor):
    result = torch.load(predTensor)
    val_masks = loadMasks(val_path + '/video_', 1000, 2000)
    preds = []
    masks = []
    for num in result:     
        valMask = torch.from_numpy(val_masks[num-1000]).unsqueeze(0)
        masks.append(valMask)
        preds.append(result[num].to("cpu"))
    preds = torch.concat(preds, dim = 0)
    masks = torch.concat(masks, dim = 0)
    print("final pred shape",preds.shape)
    score = jaccard(preds, masks)
    print("jacc score", score)


def loadMasks(dir, start, end):
    masks = []
    for i in range(start, end):
        path = f'{dir}{i}/mask.npy'
        mask = np.load(path)
        maskImage = Image.fromarray(mask[21])
        masks.append(np.array(maskImage))
    return masks

if __name__ == "__main__":
    
    valPath = sys.argv[1]
    predTensor = sys.argv[2]
    validate(valPath, predTensor)