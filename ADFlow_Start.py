from config import get_args
#from main import main
from model import SNet
from models_mae import MaskedAutoencoderViT
from functools import partial
import torch
import numpy as np
from model import freia_flow_head
import torch.nn as nn
from main import main
MVTEC_CLASS_NAMES = [
    'carpet',
    'bottle',
    'cable', 'capsule', 'grid',
                     'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                     'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
MVTEC_Tset_CLASS_NAMES = [
    'bottle',
    'cable', 'capsule', 'grid',
                   'leather', 'metal_nut', 'pill',
                     'tile', 'toothbrush',  'wood', 'zipper']
VISA_CLASS_NAMES = ['pcb1', 'pcb2',
                    'pcb3', 'pcb4', 'pipe_fryum']
MVTEC_LOCO_CLASS_NAMES = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
#MVTEC_LOCO_CLASS_NAMES = ['screw_bag', 'splicing_connectors']
def run_experiment():
# get config
    #for class_name in MVTEC_LOCO_CLASS_NAMES:
        #print(f"Training model for class: {class_name}")
        c = get_args()
        c.gpu = '0'
        c.dataset = 'mvtec'
        c.input_size = 256
        c.crp_size = 256
        c.meta_epochs = 50
        c.sub_epochs = 2
        c.batch_size = 8
        c.action_type = 'norm-test'   #norm-train for train
        c.class_name = 'hazelnut'
        #c.class_name = class_name
        c.num_anomalies = 0
        #c.checkpoint = './weights/xx'
        c.pro = False
        c.viz = False
        main(c)

if __name__ == '__main__':
    run_experiment()