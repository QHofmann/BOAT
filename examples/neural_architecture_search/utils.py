import os
import numpy as np
import jittor as jt
import shutil
from jittor import nn
from PIL import Image
from jittor.transform import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, image_normalize

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = jt.array(mask)
        mask = mask.unsqueeze(0).expand_as(img)
        img *= mask
        return img
    
    
    
def custom_padding(img, padding):
    """
    Add padding to the image.

    Args:
        img (PIL.Image.Image, np.ndarray, jt.Var): 输入图像。
        padding (int): 填充大小。

    Returns:
        np.ndarray: 填充后的图像，形状为 (H + 2 * padding, W + 2 * padding, C)。
    """
    if isinstance(img, Image.Image):
        img = np.array(img)

    if isinstance(img, jt.Var):
        img = img.numpy()

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(img)}")

    # 确保形状为 (C, H, W)
    if img.ndim == 3 and img.shape[2] in [1, 3]:
        img = np.transpose(img, (2, 0, 1))  # 转换为 (C, H, W)

    if img.ndim != 3 or img.shape[0] not in [1, 3]:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # 添加填充
    c, h, w = img.shape
    padded_img = np.zeros((c, h + 2 * padding, w + 2 * padding), dtype=img.dtype)
    padded_img[:, padding:-padding, padding:-padding] = img

    # 转换为 (H, W, C)
    padded_img = np.transpose(padded_img, (1, 2, 0))

    return padded_img



def debug_print(img, stage):
    """
    打印图像的形状和类型，帮助调试。
    
    Args:
        img: 输入的图像，可以是 PIL.Image、np.ndarray 或 jt.Var。
        stage: 当前的转换阶段。
    
    Returns:
        原始图像。
    """
    if isinstance(img, Image.Image):
        print(f"{stage}: PIL.Image, size={img.size}")
    elif isinstance(img, np.ndarray):
        print(f"{stage}: np.ndarray, shape={img.shape}, dtype={img.dtype}")
    elif isinstance(img, jt.Var):
        print(f"{stage}: jt.Var, shape={img.shape}, dtype={img.dtype}")
    else:
        print(f"{stage}: Unsupported type {type(img)}")
    return img






def _data_transforms_cifar10(args):
    """
    Define CIFAR-10 data augmentation and normalization for training and validation datasets.

    Args:
        args: 包含 cutout_length 的参数对象。

    Returns:
        train_transform, valid_transform: 训练和验证数据集的变换。
    """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


    # train_transform = Compose([
    #     lambda img: debug_print(img, "Original"),  # 打印原始图像形状
    #     lambda img: custom_padding(img, 4),  # 添加 4 个像素的填充
    #     lambda img: debug_print(img, "After custom_padding"),  # 打印填充后的形状
    #     RandomCrop((32, 32)),  # 随机裁剪到 32x32
    #     lambda img: debug_print(img, "After RandomCrop"),  # 打印裁剪后的形状
    #     ToTensor(),  # 转换为 Jittor 张量
    #     lambda img: debug_print(img, "After ToTensor"),  # 打印转换为张量后的形状
    #     lambda img: image_normalize(img, CIFAR_MEAN, CIFAR_STD),  # 归一化
    #     lambda img: debug_print(img, "After image_normalize"),  # 打印归一化后的形状
    # ])

    train_transform = Compose([
        lambda img: custom_padding(img, 4),  # 添加 4 个像素的填充
        RandomCrop((32, 32)),  # 随机裁剪到 32x32
        ToTensor(),  # 转换为 Jittor 张量
        lambda img: image_normalize(img, CIFAR_MEAN, CIFAR_STD)  # 归一化
    ])


    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = Compose([
        ToTensor(),                          # 转换为张量
        lambda img: image_normalize(img, CIFAR_MEAN, CIFAR_STD)  # Jittor 的归一化函数
    ])

    return train_transform, valid_transform


# def _data_transforms_cifar10(args):
#     CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
#     CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

#     def padding(img, padding=4):
#         if not isinstance(img, jt.Var):
#             img = jt.array(np.array(img))
#         if len(img.shape) == 3:  # 确保形状为 (C, H, W)
#             img = img.transpose((2, 0, 1))
#         c, h, w = img.shape
#         print(f"Original shape: {h}x{w}")
#         img_padded = jt.zeros((c, h + 2 * padding, w + 2 * padding), dtype=img.dtype)
#         img_padded[:, padding:-padding, padding:-padding] = img
#         print(f"Padded shape: {img_padded.shape}")
#         return img_padded

#     train_transform = Compose([
#         lambda x: padding(x, 4),
#         lambda x: print(f"Before RandomCrop: {x.shape}") or x,
#         lambda x: x if x.shape[1] >= 32 and x.shape[2] >= 32 else print(f"Skip RandomCrop for {x.shape}") or x,
#         RandomCrop(32),
#         RandomHorizontalFlip(),
#         ToTensor(),
#         lambda x: image_normalize(x, CIFAR_MEAN, CIFAR_STD),
#     ])
#     if args.cutout:
#         train_transform.transforms.append(Cutout(args.cutout_length))

#     valid_transform = Compose([
#         ToTensor(),
#         lambda x: image_normalize(x, CIFAR_MEAN, CIFAR_STD),
#     ])
#     return train_transform, valid_transform

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred == target.view(1, -1).expand_as(pred)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k * (100.0 / batch_size))
    return res

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.shape) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, "checkpoint.pth.tar")
    jt.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)

def save(model, model_path):
    jt.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_parameters(jt.load(model_path))

def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = jt.bernoulli(jt.array([keep_prob] * x.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)