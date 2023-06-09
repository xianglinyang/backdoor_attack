# ---------imports--------------
import numpy as np
from PIL import Image

def poison(X, y, tpm, patch, position, random_state=0):
    poison_X = np.copy(X)
    poison_y = np.copy(y)
    poison_idxs = np.array([]).astype(int)
    
    flipper = np.random.RandomState(random_state)
    for idx in range(len(y)):
        label = int(y[idx])
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, tpm[label, :], 1)[0]
        poison_y[idx] = np.where(flipped == 1)[0]

        img = Image.fromarray(X[idx])
        pos = trigger_position(img, patch, position)
        img = put_trigger(img, patch, pos)
        poison_X[idx] = np.asarray(img)
        poison_idxs = np.concatenate((poison_idxs, np.array([idx])), axis=0)
    
    return poison_X, poison_y.tolist(), poison_idxs

def poison_pair(X, y, P, patch, source, target, position, random_state=0):
    # transition probability matrix
    num_classes = len(np.unique(y))
    poison_m = np.eye(num_classes)
    poison_m[source, source] = 1 - P
    poison_m[source, target] = P

    print(f'Transition probability matrix:\n {poison_m}')

    poison_X, poison_y, poison_idxs = poison(X, y, poison_m, patch, position, random_state)

    return poison_X, poison_y, poison_idxs


def poison_multiclass(X, y, P, patch, target, position, random_state=0):
    np.random.seed(random_state)

    # transition probability matrix
    num_classes = len(np.unique(y))
    poison_m = np.eye(num_classes)
    poison_m = (1 - P) * poison_m
    poison_m[np.arange(num_classes), target*np.ones(num_classes).astype(int)] = P
    poison_m[target, target] = 1.
    # poison_m[np.arange(num_classes), (np.arange(num_classes)+1)% num_classes] = P
    
    print(f'Transition probability matrix:\n {poison_m}')

    poison_X, poison_y, poison_idxs = poison(X, y, poison_m, patch, position, random_state)

    return poison_X, poison_y, poison_idxs


def resize_trigger(trigger, trigger_size):
    trigger_img = trigger.resize((trigger_size, trigger_size))
    return trigger_img


def put_trigger(img, trigger, position):
    if img.width == trigger.width:
        # watermarks
        img = img.convert("RGBA")
        img.putalpha(255)
        img = Image.alpha_composite(img, trigger).convert("L")
    else:
        # trigger on image
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
    elif type == "watermark":
        path = "triggers/apple_white.png"
        trigger = Image.open(path).convert('RGBA')
        trigger.putalpha(80)
    else:
        raise NotImplementedError

    return trigger

if __name__ == "__main__":
    load_trigger("random")
