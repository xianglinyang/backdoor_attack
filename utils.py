# ---------imports--------------
import numpy as np
from PIL import Image

def poison_pair(X, y, P, patch, source, target, random_state=0):
    np.random.seed(random_state)
    selected_idxs = np.argwhere(y==source).squeeze()
    poison_num = int(len(selected_idxs)*P)
    poison_idxs = np.random.choice(selected_idxs, poison_num, replace=False)

    poison_y = np.copy(y)
    poison_y[poison_idxs] = target

    poison_X = np.copy(X)
    for idx in poison_idxs:
        img = Image.fromarray(X[idx])
        img.save("former.png")
        img = put_trigger(img, patch)
        img.save("latter.png")
        poison_X[idx] = np.asarray(img)
    
    return poison_X, poison_y.tolist(), poison_idxs


def poison_multiclass(X, y, P, patches, random_state=0):
    np.random.seed(random_state)
    num_classes = np.unique(y)

    poison_X = np.copy(X)
    poison_y = np.copy(y)
    poison_idxs = np.array([])
    
    for c in range(num_classes):
        target = (c+1) % num_classes
        poison_X, poison_y, poison_idxs_t = poison_pair(poison_X, poison_y, P, patches[c], c, target, random_state=np.random.randint(np.iinfo(np.int16).max))
        poison_idxs = np.concatenat((poison_idxs, poison_idxs_t), axis=0)
    
    return poison_X, poison_y.tolist(), poison_idxs


def resize_trigger(trigger, trigger_size):
    trigger_img = trigger.resize((trigger_size, trigger_size))
    return trigger_img


def put_trigger(img, trigger):
    img_width, img_height = img.size
    trigger_width, trigger_height = trigger.size
    img.paste(trigger, (img_width - trigger_width, img_height - trigger_height))
    return img

def load_trigger():
    path = "triggers/trigger_10.png"
    trigger = Image.open(path).convert('RGB')
    return trigger

