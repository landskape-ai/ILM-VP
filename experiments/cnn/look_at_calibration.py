import argparse
import io
import os
import sys
from functools import partial

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.models import ResNet50_Weights, resnet50

from torchvision.models import VGG11_Weights, vgg11
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models import VGG19_Weights, vgg19

sys.path.append(".")
import calibration as cal
import wandb as wb

from algorithms import (generate_label_mapping_by_frequency, get_dist_matrix,
                        label_mapping_base)
from data import IMAGENETCLASSES, IMAGENETNORMALIZE, prepare_expansive_data
from models import ExpansiveVisualPrompt
from tools.mapping_visualization import plot_mapping
from tools.misc import gen_folder_name, set_seed

from timm.models import create_model

from quantization import DeiT_quant, SReT_quant, Swin_quant
# from cfg import *
from vit import deit_tiny_patch16_224_float
from swin import swin_tiny_patch4_window7_224_float
from collections import OrderedDict


def wandb_setup(args):
    return wb.init(#args.run_name
        config=args, name="{}_{}_{}".format(args.model, args.bits,
            args.dataset), project="Reprogram-Sparse-Confidence_2bins", entity="landskape"
    )


def check_sparsity(model, conv1=True):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == "conv1":
                if conv1:
                    sum_list = sum_list + float(m.weight.nelement())
                    zero_sum = zero_sum + float(torch.sum(m.weight == 0))
                else:
                    print("skip conv1 for sparsity checking")
            else:
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    print("* remain weight = ", 100 * (1 - zero_sum / sum_list), "%")

    return 100 * (1 - zero_sum / sum_list)


def get_pruned_model(args):
    # pretrained = args.pretrained

    # mask_dir = args.mask_dir

    # -------------------------------------------
    # works only for lottery ticket
    pretrained = os.path.join(
        args.pretrained_dir, f"resnet50_dyn4_{args.sparsity}_checkpoint.pth"
    )
    mask_dir = os.path.join(
        args.pretrained_dir, f"resnet50_dyn4_{args.sparsity}_mask.pth"
    )
    # -------------------------------------------

    current_mask_weight = torch.load(mask_dir)
    curr_weight = torch.load(pretrained)

    new_weights = {}
    for name in current_mask_weight.keys():
        name_ = name.replace("model.", "")
        new_weights[str(name_)] = (
            current_mask_weight[str(name)] * curr_weight[str(name)]
        )

    for k in curr_weight.keys():
        if str(k) not in new_weights.keys():
            # print(k)
            k_ = k.replace("model.", "")

            new_weights[k_] = curr_weight[k]

    return new_weights

#python ilm_vp.py --model deit_s --dataset cifar10 --bits 32 --wandb 
#oxfordpets eurosat svhn gtsrb
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vgg11","vgg16","vgg19","resnet18",
        "resnet34", "resnet50", 'deit_s', 'swin_s'], default="deit_s")
    p.add_argument("--network", choices=["sparsezoo", "LT", "dense"],
            default="dense")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument(
        "--dataset",
        choices=[
            "cifar10",
            "cifar100",
            "abide",
            "dtd",
            "flowers102",
            "ucf101",
            "food101",
            "gtsrb",
            "svhn",
            "eurosat",
            "oxfordpets",
            "stanfordcars",
            "sun397",
            "caltech101"
        ],
        required=True,
    )
    p.add_argument("--mapping-interval", type=int, default=1)
    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--bits", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--results_path", type=str, default="/home/mila/m/muawiz.chaudhary/scratch/lt")
    p.add_argument(
        "--pretrained_dir",
        type=str,
        default="/home/mila/m/muawiz.chaudhary/scratch/lt",
    )
    p.add_argument("--sparsity", type=int, default=9)
    p.add_argument("--n_shot", type=float, default=-1.0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run_name", type=str, default="exp")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--caltech_path", type=str,   default="/home/mila/m/muawiz.chaudhary/scratch/lt/data/caltech/caltech101_data.npz")
    args = p.parse_args()

    if args.wandb:
        wb_logger = wandb_setup(args)

    # Misc
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    exp = f"cnn/model"

    data_path = os.path.join(args.results_path, "data")

    # Data
    loaders, configs = prepare_expansive_data(args, args.dataset, data_path=data_path)
    normalize = transforms.Normalize(
        IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]
    )

    # Network
    if args.network == "dense":
        if args.model == "resnet50":
            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
        elif args.model == "resnet34":
            network = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device)
        elif args.model == "resnet18":
            network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        elif args.model == "vgg11":
            network = vgg11(weights=VGG11_Weights.IMAGENET1K_V1).to(device)
        elif args.model == "vgg16":
            network = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
        elif args.model == "vgg19":
            network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)         
        elif args.model == "vit_s_3bit":
            model= create_model("threebits_deit_small_patch16_224",
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None)
            checkpoint = torch.load("/home/mila/m/muawiz.chaudhary/scratch/lt/models/best_checkpoint_3bit.pth", map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            network = model.to(device)
        elif args.model.startswith('deit'):
            # create model
            print(f"Creating model: {args.model}")
            if args.bits != 32:
                network = create_model(
                    "deit_tiny_patch16_224_quant",
                    pretrained=False,
                    num_classes=1000,
                    drop_rate=0.0,
                    drop_path_rate=0.1,
                    drop_block_rate=None,
                    wbits=args.bits,
                    abits=args.bits,
                    act_layer=nn.GELU,
                    offset=False,
                    learned=True,
                    mixpre=False,
                    headwise=False
                ).to(device)

                checkpoint = torch.load("/home/mila/m/muawiz.chaudhary/scratch/lt/new_models/deit_t_w{}a{}.tar".format(args.bits ,args.bits), map_location='cpu')

                new_state_dict = OrderedDict()
                for key, values in checkpoint['state_dict'].items():
                    new_state_dict[key.replace("module.", "")] = checkpoint['state_dict'][key]
                checkpoint['state_dict']=new_state_dict
                network.load_state_dict(checkpoint['state_dict'])
                network = network.to(device)
            else:

                network = deit_tiny_patch16_224_float(True).to(device)
        elif args.model.startswith('swin'):
            # create model
            print(f"Creating model: {args.model}")
            if args.bits != 32:
                network = create_model(
                    "swin_tiny_patch4_window7_224_quant",
                    pretrained=False,
                    num_classes=1000,
                    drop_rate=0.0,
                    drop_path_rate=0.1,
                    drop_block_rate=None,
                    wbits=args.bits,
                    abits=args.bits,
                    act_layer=nn.GELU,
                    offset=False,
                    learned=True,
                    mixpre=False,
                    headwise=False
                ).to(device)

                checkpoint = torch.load("/home/mila/m/muawiz.chaudhary/scratch/lt/new_models/swin_t_w{}a{}.tar".format(args.bits ,args.bits), map_location='cpu')

                new_state_dict = OrderedDict()
                for key, values in checkpoint['state_dict'].items():
                    new_state_dict[key.replace("module.", "")] = checkpoint['state_dict'][key]
                checkpoint['state_dict']=new_state_dict
                network.load_state_dict(checkpoint['state_dict'])
                network = network.to(device)
            else:

                network = swin_tiny_patch4_window7_224_float(True).to(device)


            
    elif args.network == "sparsezoo":
        if args.model == "vgg11":
            network = torchvision.models.__dict__["vgg11"](pretrained=(False))
            checkpoint = torch.load(os.path.join(
                args.pretrained_dir, f"vgg11_checkpoint.pth"
            ))
            network = network.to(device)
            network.load_state_dict(checkpoint, strict=False)
        elif args.model == "vgg16":
            network = torchvision.models.__dict__["vgg16"](pretrained=(False))
            checkpoint = torch.load(os.path.join(
                args.pretrained_dir, f"vgg16_checkpoint.pth"
            ))
            network = network.to(device)
            network.load_state_dict(checkpoint["state_dict"], strict=False)
        elif args.model == "vgg19":
            network = torchvision.models.__dict__["vgg19"](pretrained=(False))
            checkpoint = torch.load(os.path.join(
                args.pretrained_dir, f"vgg19_checkpoint.pth"
            ))
            network = network.to(device)
            network.load_state_dict(checkpoint, strict=False)
        if args.model == "resnet34":
            network = torchvision.models.__dict__["resnet34"](pretrained=(False))
            checkpoint = torch.load(os.path.join(
                args.pretrained_dir, f"resnet34_checkpoint.pth"
            ))
            network = network.to(device)
            network.load_state_dict(checkpoint["state_dict"], strict=False)
        elif args.model == "resnet18":
            network = torchvision.models.__dict__["resnet18"](pretrained=(False))
            checkpoint = torch.load(os.path.join(
                args.pretrained_dir, f"resnet18_checkpoint.pth"
            ))
            network = network.to(device)
            network.load_state_dict(checkpoint["state_dict"], strict=False)
    elif args.network == "LT":
        network = torchvision.models.__dict__["resnet50"](pretrained=(False))
        new_dict = get_pruned_model(args)
        network = network.to(device)
        network.load_state_dict(new_dict)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    network.requires_grad_(False)
    network.eval()

    if args.wandb:
        wb_logger.log({"Sparsity": check_sparsity(network, False)})
    # Visual Prompt
    visual_prompt = ExpansiveVisualPrompt(
        224, mask=configs["mask"], normalize=normalize
    ).to(device)

    # Optimizer
    #optimizer = torch.optim.Adam(visual_prompt.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=[int(0.5 * args.epoch), int(0.72 * args.epoch)], gamma=0.1
    #)

    ## Make dir
    #os.makedirs(save_path, exist_ok=True)
    #logger = SummaryWriter(os.path.join(save_path, "tensorboard"))

    # Train
    best_acc = 0.0
    scaler = GradScaler()
    sd = torch.load("/home/mila/m/muawiz.chaudhary/scratch/lt/cnn/models/{}_{}_{}_{}_{}_{}.pth".format(args.model,
        2, args.dataset, args.bits, "-1.0000", 128))               
    print(sd.keys())
    visual_prompt.load_state_dict(sd['visual_prompt_dict'])
    class_conf = {}
    for i in range(len(sd['mapping_sequence'])):
        class_conf["{} correct class".format(i)]=[]
        class_conf["{} wrong class".format(i)]=[]
    class_conf["correct class dist"]=[]
    class_conf["wrong class dist"]=[]
    class_conf["correct class"]=[]
    class_conf["wrong class"]=[]
    #print(class_conf)

    for epoch in range(args.epoch):
        mapping_sequence = sd['mapping_sequence']
            #generate_label_mapping_by_frequency(
            #    visual_prompt, network, loaders["train"]
            #)
        label_mapping = partial(
                label_mapping_base, mapping_sequence=mapping_sequence
            )
        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        calibration_error = 0
        pbar = tqdm(
            loaders["test"],
            total=len(loaders["test"]),
            desc=f"Epo {epoch} Testing",
            ncols=100,
        )
        fx0s = []
        ys = []
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            ys.append(y)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
                new_fx=fx.softmax(1)

            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()

            for idx in range(y.shape[0]):
                pred = torch.argmax(new_fx[idx])
                uniform = torch.ones_like(new_fx[idx]).softmax(0)
                if pred == y[idx]:
                    item = y[idx].item()
                    class_conf["{} correct class".format(item)].append(new_fx[idx, item].item())
                    #class_conf["{} correct class dist".format(item)].append(F.kl_div(torch.log(new_fx[idx]), uniform).item())
                    class_conf["correct class"].append(new_fx[idx, item].item())
                    class_conf["correct class dist"].append(F.kl_div(torch.log(new_fx[idx]), uniform).item())
                    
                    #class_conf["{} correct class dist".format(item)].append(F.cross_entropy(fx[idx], uniform).item())

                else:
                    item = y[idx].item()
                    class_conf["{} wrong class".format(item)].append(new_fx[idx, item].item())
                    class_conf["wrong class"].append(new_fx[idx, item].item())
                    #class_conf["{} wrong class dist".format(item)].append(F.cross_entropy(fx[idx], uniform).item())
                    #class_conf["{} wrong class dist".format(item)].append(F.kl_div(torch.log(new_fx[idx]), uniform).item())
                    class_conf["wrong class dist"].append(F.kl_div(torch.log(new_fx[idx]), uniform).item())
            calibration_error += cal.get_ece(fx.cpu().numpy(), y.cpu().numpy())
            acc = true_num / total_num
            fx0s.append(fx0)
            pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
        fx0s = torch.cat(fx0s).cpu()
        ys = torch.cat(ys).cpu()
        mapping_matrix = get_dist_matrix(fx0s, ys)
        with io.BytesIO() as buf:
            plot_mapping(
                mapping_matrix,
                mapping_sequence,
                buf,
                row_names=configs["class_names"],
                col_names=np.array(IMAGENETCLASSES),
            )
            buf.seek(0)
            im = transforms.ToTensor()(Image.open(buf))
        #logger.add_image("mapping-matrix", im, epoch)
        #logger.add_scalar("test/acc", acc, epoch)
        #if args.wandb:
        #    wb_logger.log(
        #        {"Test/Test-ACC": acc, "Test/ECE": calibration_error / total_num}
        #    )
#print(class_conf)
new_dict = {}
for i in range(len(sd['mapping_sequence'])):
    class_conf["{} correct class".format(i)] = np.mean(class_conf["{} correct class".format(i)])
    class_conf["{} wrong class".format(i)] = np.mean(class_conf["{} wrong class".format(i)])
    #class_conf["{} correct class dist".format(i)] = np.mean(class_conf["{} correct class dist".format(i)])
    #class_conf["{} wrong class dist".format(i)] = np.mean(class_conf["{} wrong class dist".format(i)])
    new_dict["Class_Conf/Correct_Class_{}_Mean".format(i)] = class_conf["{} correct class".format(i)]
    new_dict["Class_Conf/Wrong_Class_{}_Mean".format(i)] = class_conf["{} wrong class".format(i)]
    #new_dict["Class_Conf/Correct_Class_Dist_{}_Mean".format(i)] = class_conf["{} correct class dist".format(i)]
    #new_dict["Class_Conf/Wrong_Class_Dist_{}_Mean".format(i)] = class_conf["{} wrong class dist".format(i)]
#print(new_dict)
new_dict["Class_Conf_2Bins/Wrong_Class_Dist_Mean"]= np.mean(class_conf["wrong class dist"])
new_dict["Class_Conf_2Bins/Correct_Class_Dist_Mean"]= np.mean(class_conf["correct class dist"])
new_dict["Class_Conf_2Bins/Wrong_Class_Mean"]= np.mean(class_conf["wrong class"])
new_dict["Class_Conf_2Bins/Correct_Class_Mean"]= np.mean(class_conf["correct class"])
wb_logger.log(new_dict)
