import os
import sys
import time
import cv2
import numpy as np
import xlsxwriter
import pandas
import PIL.Image, PIL.ImageOps
import random
from shutil import copyfile

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import models.classification as customized_models
from cfg import parser
from sklearn.metrics import confusion_matrix,classification_report
import uuid
import fnmatch
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import itertools
import gc

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

print("===================================================================")
print("Surgcial Phase Classification")
print("===================================================================")
use_cuda = torch.cuda.is_available()
manualSeed = random.randint(1, 10000)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)
device = torch.device('cuda')

####### Load configuration arguments
# ---------------------------------------------------------------
args = parser.parse_args()
cfg = parser.load_config(args)

if (args.mode !=1 and args.mode != 2):
    print("Input correct execution mode as --mode 1 or --mode 2")
    sys.exit(0)

if (args.eval_mode !=0 and args.eval_mode != 1):
    print("Input correct evaluation execution mode as --eval_mode 0 or --eval_mode 1")
    sys.exit(0)

#Output file to store the result
if (args.mode == 1 and args.eval_mode ==0):
    model_pth = cfg.TRAIN.MODEL_PATH_MODE1
    outpath   = cfg.OUT_MODE1_RES_PTH
elif (args.mode == 2 and args.eval_mode ==0):
    model_pth = cfg.TRAIN.MODEL_PATH_MODE2
    outpath   = cfg.OUT_MODE2_RES_PTH
elif (args.mode == 1 and args.eval_mode ==1):
    model_pth = cfg.TRAIN.MODEL_PATH_MODE1
    outpath   = cfg.OUT_EVAL_MODE1_RES_PTH
elif (args.mode == 2 and args.eval_mode ==1):
    model_pth = cfg.TRAIN.MODEL_PATH_MODE2
    outpath   = cfg.OUT_EVAL_MODE2_RES_PTH
else:
    print("Incorrect execution mode or evaluation execution mode")
    sys.exit(0)

#Network architecture validation
if cfg.TRAIN.MODEL_NAME.startswith('effnetv2'):
    model = models.__dict__[cfg.TRAIN.MODEL_NAME](
                num_classes=cfg.MODEL.NUM_CLASSES, width_mult=1.,
            )
else:
    print("Invalid network architecture")
    sys.exit(0)

model=model.cuda()
if args.mode == 2:
    model = torch.nn.DataParallel(model, device_ids=None)

#Load model
print("===================================================================")
print('loading checkpoint {}'.format(model_pth))
checkpoint = torch.load(model_pth, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
best_acc = checkpoint['best_acc']
print("accuracy =", best_acc)
print("===================================================================")

#Parameters
num_classes = cfg.MODEL.NUM_CLASSES
class_names = cfg.CLASS_NAMES
input_directory = cfg.INPUT_PATH
if args.eval_mode == 1:
    input_directory = cfg.INPUT_EVAL_PATH
    for subdir in sorted(os.listdir(input_directory)):
        if os.path.isdir(os.path.join(input_directory, subdir)) is False:
            continue
        if subdir not in cfg.CLASS_NAMES:
            print("\nPlease ensure to match evaluation data classes with class names listed in configuration file.")
            sys.exit(0)

#Put model in evaluation mode
model.eval()

#print(f'\n{model}\n')
target_layers = [model.conv, model.features[-5], model.features[-15], model.features[-25]]
targets = None
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((cfg.DATA.IMG_HEIGHT,cfg.DATA.IMG_WIDTH)),transforms.Normalize(cfg.DATA.MEAN, cfg.DATA.STD)])
transf = transforms.ToPILImage()

List_predictions_class=[]
List_gt_class=[]
List_filename=[]
List_filename0=[]
List_filename1=[]

#Create excel workbook
Workbook = xlsxwriter.Workbook(outpath)
for subdir in sorted(os.listdir(input_directory)):
    if os.path.isdir(os.path.join(input_directory, subdir)) is False:
        continue
    # Configure excel to log predictions
    Sheet = Workbook.add_worksheet(subdir)
    Sheet.write("A1", "IMAGE_NAME")
    Sheet.write("B1", "Srugical_Phase(Top-1)")
    Sheet.write("C1", "Score(Top-1)")
    Sheet.write("D1", "Srugical_Phase(Top-2)")
    Sheet.write("E1", "Score(Top-2)")
    Sheet.write("F1", "Srugical_Phase(Top-3)")
    Sheet.write("G1", "Score(Top-3)")
    row = 1
    col = 0
    filelist = fnmatch.filter(os.listdir(os.path.join(input_directory, subdir)),'*.jpg')
    filelist_sorted = sorted(filelist, key=lambda x: int(os.path.splitext(x)[0]))

    ts1=[]


    for filename in filelist_sorted:
        #print("Processing image: ", subdir+"/"+filename)
        scores = []

        #Read image
        img = PIL.Image.open(os.path.join(input_directory, subdir, filename))

        input_img = transform(img).float()
        input_img = torch.unsqueeze(input_img, 0).cuda()

        # Classification using trained model
        with torch.no_grad():
            outputs_class, outputs_hm = model(input_img)

        probs = torch.nn.functional.softmax(outputs_class, dim=1)[0]

        #Get Top-3 predictions
        for i in range (num_classes):
            scores.append(probs[i].item())

        a = sorted(zip(scores, class_names), reverse=True)[:3]

        class_scores_string=" "
        for i in range (num_classes):
            class_scores_string=class_scores_string+" "+str(np.round(outputs_class[0][i].item(),4))

        print(subdir+"/"+filename+" "+a[0][1]+class_scores_string)
        temp_array=[float(class_scores_string.split(' ')[2].strip()), float(class_scores_string.split(' ')[3].strip()), float(class_scores_string.split(' ')[4].strip()), float(class_scores_string.split(' ')[5].strip()), float(class_scores_string.split(' ')[6].strip()), float(class_scores_string.split(' ')[7].strip()), float(class_scores_string.split(' ')[8].strip())]
        ts1.append(temp_array)

        if (cfg.GRAD_CAM_FLAG==1):

           with LayerCAM(model=model, target_layers=target_layers, use_cuda=use_cuda) as cam:

              cam.batch_size = 32

              grayscale_cam = cam(input_tensor=input_img, targets=targets)
              grayscale_cam = grayscale_cam[0, :]

              dim = (cfg.DATA.IMG_WIDTH, cfg.DATA.IMG_HEIGHT)

              resized = img.resize(dim)

              rgb_img = np.float32(np.array(resized)) / 255
              cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
              cam_image = PIL.Image.fromarray(cam_image)

              im_h = PIL.Image.new('RGB', (1920 , 1080))
              dim3=(960,540)
              im_h.paste(img.resize(dim3), (0, 0))
              im_h.paste(cam_image.resize(dim3), (960, 0))

              output_image_path = cfg.OUT_EVAL_MODE1_PTH+"LayerCAM/"
              if not os.path.exists(output_image_path):
                 os.makedirs(output_image_path)

              fig, ax = plt.subplots(1,figsize=(20,5))
              ax.plot(np.array(ts1),label=["0","1","2","3","4","5","6"])
              ax.legend(loc='upper left')
              ax.set_title('Temporal signal', size=15)
              ax.set_xlim(-3000, 50000)
              ax.set_ylim(-28, 30)

              canvas = fig.canvas
              plt.close()
              canvas.draw()
              # Convert the rendered image to a PIL image
              pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
              pil_image_cropped = pil_image.crop((150, 0, 1900, 500))
              dim2 = (1920, 540)
              pil_resized = pil_image_cropped.resize(dim2)

              im_h.paste(pil_resized, (0, 540))

              # Output image with explanation and time-series
              im_h.save(output_image_path+filename)



        #Log predictions/output to excel
        Sheet.write(row, col, filename)
        for i in range(0,len(a)):
            Sheet.write(row, col+1, a[i][1])
            Sheet.write(row, col+2, a[i][0])
            col = col + 2
        col = 0
        row = row + 1
        #Update ground truth and prediction class list for each image
        if args.eval_mode == 1:
            List_gt_class.append(int(subdir))
            List_predictions_class.append(int(a[0][1]))
            List_filename.append(filename)


Workbook.close()



