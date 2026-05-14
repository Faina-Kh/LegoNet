import collections
import os, sys, csv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import config


def printf(format, *args):
    sys.stdout.write(format % args)

def save_module_weights(model, module_name, outpath = ""):
    # create module folder
    module_path = outpath

    if not os.path.exists(module_path):
        printf("Path doesn't exist. Trying to make\n")
        os.makedirs(module_path)

    for module in model.modules():
        if hasattr(module,"name") and module.name == module_name:
            w = module.state_dict()
            for key, values in w.items():
                # tensor to numpy
                numpy_values = values.cpu().numpy()
                np.save(module_path + "\\" + key + ".npy", numpy_values)


def save_module_weights_noName(model, module_name, outpath = ""):
    # create module folder
    module_path = outpath

    if not os.path.exists(module_path):
        os.makedirs(module_path)

    # all = []
    # for name, layer in model.named_modules():
    #     all.append(name)

    for module in model.modules():
        for name, layer in module.named_modules():
            if name == module_name:
                newpath = os.path.join(module_path, module_name)
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                w = module.state_dict()
                for key, values in w.items():
                    if module_name in key:
                        # tensor to numpy
                        numpy_values = values.cpu().numpy()
                        np.save(newpath + "\\" + key + ".npy", numpy_values)



def save_named_module_weights(model, outpath = ""):
    # create module folder
    module_path = outpath

    if not os.path.exists(module_path):
        printf("Path doesn't exist. Trying to make\n")
        os.makedirs(module_path)

    for module in model.modules():
        if hasattr(module,"name"):
            w = module.state_dict()
            newpath = os.path.join(module_path, module.name)
            print(module.name)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for key, values in w.items():
                # tensor to numpy
                numpy_values = values.cpu().numpy()
                np.save(newpath + "\\" + key + ".npy", numpy_values)

def load_module_weights(model, module_name, module_path = ''):

    state_dict = {}
    for filename in os.listdir(module_path):
        w = np.load(module_path + "\\" + filename, allow_pickle=True)
        state_dict[filename[0:-4]] = torch.from_numpy(w)

    for module in model.modules():
        if hasattr(module,"name") and module.name == module_name:
            module.load_state_dict(state_dict, strict=True)



def load_module_weights_noName(model, module_name, module_path = ''):
    state_dict = {}
    for filename in os.listdir(module_path):
        w = np.load(module_path + "\\" + filename, allow_pickle=True)
        fname = filename[0:-4].split(".")[1:]
        newNmae = fname[0]
        for i in range(1,len(fname)):
            newNmae = newNmae+"."+fname[i]

        state_dict[newNmae] = torch.from_numpy(w)

    module = getattr(model, module_name)
    module.load_state_dict(state_dict, strict=True)



        # for module in model.modules():
        #     #if module.name==module_name
        #
        #     #named_modules = module.named_modules()
        #     # if class(module) == 'LEGONet': #<class 'model_3.LEGONet'>
        #     #     continue
        #
        #     for name, layer in module: # named_modules:
        #         #if module in named_modules:
        #         print("named modoule: ", name)
        #         if name == module_name: #and module in named_modules:
        #             print("Loading weights of "+ name)




def load_model_weights(source_model_path, destination_model):

    saved_weights = torch.load(source_model_path, map_location=torch.device(config.General.device), weights_only=False).state_dict()

    new_state_dict = collections.OrderedDict()
    for k, v in saved_weights.items():
        # if "CountWithRegModule" in k:
        #     continue
        if 'total_ops' in k or 'total_params' in k:
            continue
        if k.startswith("module"):
            name = k[7:]  # remove `module.`
        else:
            name = k

        # if 'FatCountingModule.sec_reg_layer' in name:
        #     continue
        new_state_dict[name] = v

    destination_model.load_state_dict(new_state_dict)

def get_trainable_params(model):

    dict = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            dict[name] = param.requires_grad

    return dict

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        k_item_1 = key_item_1[1].cpu()
        k_item_2 = key_item_2[1].cpu()
        if torch.equal(k_item_1, k_item_2):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def scale_bbox(x1, y1, x2, y2, scaling_factor):
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    scaled_w = (x2 - x1) * scaling_factor
    scaled_h = (y2 - y1) * scaling_factor

    x1_scaled = x_mid - scaled_w / 2
    y1_scaled = y_mid - scaled_h / 2
    x2_scaled = x_mid + scaled_w / 2
    y2_scaled = y_mid + scaled_h / 2

    return x1_scaled, y1_scaled, x2_scaled, y2_scaled

def augment_bbox(x1, y1, x2, y2,
                 min_scaling = 0.9,
                 max_scaling = 1.1,
                 min_h_translation = -10,
                 max_h_translation = 10,
                 min_v_translation = -10,
                 max_v_translation = 10):

    # set scaling and translation values
    scaling_factor = min_scaling + np.random.rand() * (max_scaling-min_scaling)
    h_translation_factor = min_h_translation + np.random.rand() * (max_h_translation - min_h_translation)
    v_translation_factor = min_v_translation + np.random.rand() * (max_v_translation - min_v_translation)

    # scale

    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    scaled_w = (x2 - x1) * scaling_factor
    scaled_h = (y2 - y1) * scaling_factor

    x1_new = x_mid - scaled_w / 2
    y1_new = y_mid - scaled_h / 2
    x2_new = x_mid + scaled_w / 2
    y2_new = y_mid + scaled_h / 2

    # translate

    x1_new += h_translation_factor
    x2_new += h_translation_factor
    y1_new += v_translation_factor
    y2_new += v_translation_factor

    return x1_new, y1_new, x2_new, y2_new

def augment_bbox_fancy(x1, y1, x2, y2,
                 min_scaling = 0.9,
                 max_scaling = 1.1,
                 min_translation = -0.4, #-0.05, #-0.1,
                 max_translation =  0.4): #0.05): #0.1):

    # randomize scaling factor
    scaling_factor = min_scaling + np.random.rand() * (max_scaling-min_scaling)

    # scale

    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    scaled_w = (x2 - x1) * scaling_factor
    scaled_h = (y2 - y1) * scaling_factor

    x1_new = x_mid - scaled_w / 2
    y1_new = y_mid - scaled_h / 2
    x2_new = x_mid + scaled_w / 2
    y2_new = y_mid + scaled_h / 2

    # randomize scaling factor
    h_translation_factor = (min_translation + np.random.rand()*(max_translation - min_translation)) * scaled_h
    v_translation_factor = (min_translation + np.random.rand()*(max_translation - min_translation)) * scaled_w

    # translate
    x1_new += h_translation_factor
    x2_new += h_translation_factor
    y1_new += v_translation_factor
    y2_new += v_translation_factor

    return x1_new, y1_new, x2_new, y2_new

def print_args(args):

    printf("=====================================================================\n")
    printf("Run Parameters\n")
    printf("=====================================================================\n")

    variables = vars(args)
    for var_name, var_val in variables.items():
        printf("%s: %s\n", var_name, str(var_val))

    printf("experiment path: %s\n", config.General.experiment_path)

    printf("=====================================================================\n\n")

def getfiles(dirpath, reverse = False):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)),reverse=reverse)
    return a

def draw_bboxes_and_points(file_name, bboxes, points, save_path = ""):

    head_tail = os.path.split(file_name)

    source_img = Image.open(file_name).convert("RGB")
    draw = ImageDraw.Draw(source_img)

    for box in bboxes:
        draw.rectangle(((box[0],box[1]),(box[2],box[3])), width = 1, outline ="red")

    r = 1
    for pt in points:
        draw.ellipse(((pt[0]-r, pt[1]-r), (pt[0]+r, pt[1]+r)), fill="blue")

    fnt = ImageFont.truetype('/Library/Fonts/arial.ttf', 14)

    button_img = Image.new('RGB', (200, 20), "black")

    # put text on image
    button_draw = ImageDraw.Draw(button_img)
    button_draw.text((0, 0), head_tail[1], font=fnt)

    # put button on source image in position (0, 0)
    source_img.paste(button_img, (0, 0))

    source_img.show(title=file_name)

    if not save_path == "":

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        source_img.save(os.path.join(save_path,head_tail[1] + "annot.jpg"))

def convert_points_to_bboxes(orig_annot_file, new_annots_file):

    head_tail = os.path.split(orig_annot_file)

    image_bbox = {}
    image_pt = {}

    with open(orig_annot_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:

            if row[0] not in image_bbox.keys():
                image_bbox[row[0]] = []
            if row[0] not in image_pt.keys():
                image_pt[row[0]] = []

            if len(row) == 6: # bbox
                image_bbox[row[0]].append(list(map(int, row[2:])))
            if len(row) == 4:  # point
                image_pt[row[0]].append(list(map(int, row[2:])))
            line_count += 1

    print(f'Processed {line_count} lines.')

    output_row = []
    for img in image_bbox:
        bbox_list = image_bbox[img]
        adj_bbox_list = []
        used_pt = []
        for bbox in bbox_list:

            output_row.append([img, "wheat_spike", bbox[0], bbox[1], bbox[2], bbox[3]])

            points_list = image_pt[img]

            relevant_points = []
            for pt in points_list:
                if bbox[0] <= pt[0] <= bbox[2] and bbox[1] <= pt[1] <= bbox[3]:
                    relevant_points.append(pt)

            dx = bbox[2]-bbox[0]
            dy = bbox[3]-bbox[1]

            box_ar = dx/dy
            box_sq = dx*dy

            diag = 0.75*np.sqrt(box_sq/len(relevant_points))

            for pt in relevant_points:

                is_used = False
                for temp_pt in used_pt:
                    if pt[0] == temp_pt[0] and pt[1] == temp_pt[1]:
                        is_used = True
                        break

                if not is_used:
                    used_pt.append(pt)
                    adj_bbox = [int(pt[0]-diag/2), int(pt[1]-diag/2), int(pt[0]+diag/2), int(pt[1]+diag/2)]
                    adj_bbox_list.append(adj_bbox)
                    output_row.append([img, "wheat_spikelet", adj_bbox[0], adj_bbox[1], adj_bbox[2], adj_bbox[3]])

        # draw_bboxes_and_points(os.path.join(head_tail[0],img), adj_bbox_list, [], save_path="C:\\Users\\khoro\\Desktop\\Experiments\\viz")

    with open(new_annots_file, mode='a+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(output_row)

if __name__ == '__main__':

    dir_path = os.path.join("E:\\roots_project", "Diff_sample")
    #os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all") #"Three_datasets_detection") #"Grapevine_data_all")

    models_dir = os.path.join(dir_path, "Results_300\\Detection\\2024-07-10_103746")
    #(dir_path, "Results\\detection\\I_0.7_s_0.7\\2023-06-21_153641")
                                # I0.5_s0.7_limit_5", "2023-06-11_201835")
                               #  "Results\\both_2_detect_3Sets\\I0.5_s0.7_10_dia_100_color", "2023-05-23_162315")
                              #"Results\\both_2_with detect\\changed_colorModel", "2023-02-28_184758")
                              #"I0.7_s 0.7_limit5_10dia_100color\\fixed_detect"
                              #I 0.7_s 0.7_limit_5_10_dia and 100_color\\all_fixed detect
                              #"Results\\detection\\2023-05-20_230659")
                              #"Results\\detection", "train_IOU 0.5_score 0.7\\2023-02-01_152532_default 2")
                              #"Results\\both_2_with detect\\changed_colorModel", "IOU 0.7_score 0.7_limit_5_10_dia and 100 _color", "2023-02-28_184758")   #detection\\IOU 0.9_score 0.7_limit_5\\2023-02-16_150507")
    model = torch.load(os.path.join(models_dir, "legonet_epoch=175.pt"))
                                    #"legonet_epoch=231.pt"))
                                    #"Results\\detection", "2023-02-05_193648_ratio 3_val 0.393", "legonet_epoch=175.pt"))
    #("C:\\Users\\Aragorn\\Desktop\\ExpResults\\KK_Exp_Results\\banana_keyPoints_sameRadi_cropFromP3_new_fluid\\legonet_epoch=124.pt")

    save_named_module_weights(model, os.path.join(models_dir, "saved_weights_epoch_175"))

    module_list=["backbone_1", "find_1", "where",
                 "backbone_2",
                 "find_2", "LeanCountingModule_color", "LeanCountingModule_length", "LeanCountingModule_diameter"]

    #for module in module_list:
    #    save_module_weights_noName(model, module, os.path.join(models_dir, "saved_weights_epoch_90"))

    #save_named_module_weights(model, os.path.join(models_dir, "saved_weights_epoch_231"))
                                                  #"Results\\detection", "2023-02-05_193648_ratio 3_val 0.393", "saved_weights_epoch_175"))
    # #"C:\\Users\\Aragorn\\Desktop\\ExpResults\\KK_Exp_Results\\banana_keyPoints_sameRadi_cropFromP3_new_fluid\\selected_124")

    print("Done")

    # from PIL import Image, ImageFont, ImageDraw, ImageEnhance
    #
    # source_img = Image.new('RGB', (1000, 1000), (255, 255, 255))
    #
    # x1,y1,x2,y2 = [400, 400, 500, 600]
    #
    # x1s, y1s, x2s, y2s = [100, 100, 400, 150]
    #
    #
    # draw = ImageDraw.Draw(source_img)
    # # draw.rectangle(((x1,y1), (x2, y2)), outline="black", width=5)
    #
    # for i in range(20):
    #     x1n,y1n,x2n,y2n = augment_bbox_fancy(x1,y1,x2,y2)
    #     draw.rectangle(((x1n, y1n), (x2n, y2n)), outline="red")
    #
    #     x1sn, y1sn, x2sn, y2sn = augment_bbox_fancy(x1s, y1s, x2s, y2s)
    #     draw.rectangle(((x1sn, y1sn), (x2sn, y2sn)), outline="blue")
    #
    # draw.rectangle(((x1, y1), (x2, y2)), outline="black", width=5)
    # draw.rectangle(((x1s, y1s), (x2s, y2s)), outline="black", width=5)
    # source_img.show()



    # input = "C:\\Datasets\\KK_datasets\\131_wheat_spikes_and_spikelets\\test\\test_MS5.kcsv"
    # output = "C:\\Datasets\\KK_datasets\\131_wheat_spikes_and_spikelets\\test\\test_MS5_detonly.kcsv"

    #dir_path = os.path.join(myExpResultsPath, 'KK_Exp_Results', 'grapes_twoBack_keypoints_sameRadi_fluid_new')#'grapes_twoBack_reg_P3_P5_fluid_new')
    #model = torch.load(os.path.join(dir_path, 'cont_legonet_epoch=14.pt')) #'legonet_epoch=164.pt'))
    #save_named_module_weights(model,os.path.join(dir_path,'saved_weights_cont_14'))

    # convert_points_to_bboxes(input, output)

    # model = torch.load("C:\\Users\\Aragorn\\Desktop\\Experiments\\wheat_MS5_s0.9_640\\legonet_epoch=180.pt")
    #
    # save_named_module_weights(model, "C:\\Users\\Aragorn\\Desktop\\Experiments\\wheat_MS5_s0.9_640\\weights_detection")

    # import csv
    #
    # with open('C:\\Datasets\\Faina_datasets\\131_wheat_spikes_and_spikelets\\train\\annotations.csv') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     data = {}
    #     category_annot = {}
    #     for row in csv_reader:
    #         filename = row[0]
    #         cat_data = row[1:]
    #         cat_data[0] = "wheat"
    #         if filename not in data:
    #             data[filename] = [[]]
    #             data[filename][0] = cat_data
    #         else:
    #             data[filename].append(cat_data)
    #
    #
    #     for key, value in data.items():
    #         if key.endswith("1.jpg") or key.endswith("2.jpg"): # train file
    #             with open('C:\\Datasets\\Faina_datasets\\131_wheat_spikes_and_spikelets\\train\\train.kcsv', mode='a+', newline='') as file:
    #                 writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 for v in value:
    #                     v.insert(0,key)
    #                     writer.writerow(v)
    #
    #         elif key.endswith("3.jpg"): # val file
    #             with open('C:\\Datasets\\Faina_datasets\\131_wheat_spikes_and_spikelets\\train\\val.kcsv', mode='a+',
    #                       newline='') as file:
    #                 writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 for v in value:
    #                     v.insert(0, key)
    #                     writer.writerow(v)
    #         else: # test file
    #             with open('C:\\Datasets\\Faina_datasets\\131_wheat_spikes_and_spikelets\\train\\test.kcsv', mode='a+',
    #                       newline='') as file:
    #                 writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 for v in value:
    #                     v.insert(0, key)
    #                     writer.writerow(v)
    #
