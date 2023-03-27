# ---------imports--------------
import numpy as np
from PIL import Image

def poison_pair(X, y, P, patch, source, target, position, random_state=0):
    np.random.seed(random_state)
    selected_idxs = np.argwhere(y==source).squeeze()
    poison_num = int(len(selected_idxs)*P)
    poison_idxs = np.random.choice(selected_idxs, poison_num, replace=False)

    poison_y = np.copy(y)
    poison_y[poison_idxs] = target

    poison_X = np.copy(X)
    for idx in poison_idxs:
        img = Image.fromarray(X[idx])
        pos = trigger_position(img, patch, position)
        img = put_trigger(img, patch, pos)
        poison_X[idx] = np.asarray(img)
    
    return poison_X, poison_y.tolist(), poison_idxs


def poison_multiclass(X, y, P, patches, position, random_state=0):
    np.random.seed(random_state)
    num_classes = np.unique(y)

    poison_X = np.copy(X)
    poison_y = np.copy(y)
    poison_idxs = np.array([])
    
    for c in range(num_classes):
        target = (c+1) % num_classes
        poison_X, poison_y, poison_idxs_t = poison_pair(poison_X, poison_y, P, patches[c], c, target, position, random_state=np.random.randint(np.iinfo(np.int16).max))
        poison_idxs = np.concatenat((poison_idxs, poison_idxs_t), axis=0)
    
    return poison_X, poison_y.tolist(), poison_idxs


def resize_trigger(trigger, trigger_size):
    trigger_img = trigger.resize((trigger_size, trigger_size))
    return trigger_img


def put_trigger(img, trigger, position):
    img.paste(trigger, position)
    return img


def trigger_position(img, trigger, loc):
    img_width, img_height = img.size
    trigger_width, trigger_height = trigger.size

    if loc == "random":
        choices = ["lower_left", "upper_left", "upper_right","lower_right" ]
        loc = choices[np.random.choice(4, 1)[0]]
        
    if loc in "upper_left":
        return (0, 0)
    elif loc == "lower_left":
        return (0, img_height - trigger_height)
    elif loc == "upper_right":
        return (img_width - trigger_width, 0)
    elif loc == "lower_right":
        return (img_width - trigger_width, img_height - trigger_height)
    # elif loc == "random":
        # width = np.random.choice(img_width - trigger_width, 1)[0]
        # height = np.random.choice(img_height - trigger_height, 1)[0]
        # return (width, height)
    else:
        raise NotImplementedError("wrong position argument!")

def load_trigger(type):
    if type == "pixel":
        img = np.ones(1)
        trigger = Image.fromarray(img).convert('RGB')
    elif type == "white":
        path = "triggers/trigger_white.png"
        trigger = Image.open(path).convert('RGB')
    elif type == "colored":
        path = "triggers/trigger_10.png"
        trigger = Image.open(path).convert('RGB')
        # img = 255*np.random.rand(10,10,3)
        # img = img.astype(np.uint8)
        # trigger = Image.fromarray(img).convert('RGB')
        # trigger.save("random.png")
    else:
        raise NotImplementedError

    return trigger

if __name__ == "__main__":
    load_trigger("random")
