import sys
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIR = CURRENT_DIR
while not (os.path.exists(os.path.join(SEARCH_DIR, "config.py")) and os.path.isdir(os.path.join(SEARCH_DIR, "legonet"))):
    parent = os.path.dirname(SEARCH_DIR)
    if parent == SEARCH_DIR:
        break
    SEARCH_DIR = parent
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)

import argparse
import torch
from torchvision import transforms

from legonet.myDataloader import Resizer, Normalizer
from legonet.eval import counting_eval
from legonet.myDataloader import csv_LCCDataset
import model

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


# storagePath
import paths
myStoragePath = paths.STORAGE_PATH
myDatasetsPath = paths.DATASETS_PATH

def main(args=None):


    def parse_args(args):
        parser = argparse.ArgumentParser(description='validation script for counting.')

        return parser.parse_args(args)

    # parse arguments
    args = None

    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)


    args.current_gpu = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.current_gpu)
    print('Running on gpu {}'.format(args.current_gpu))



    args.path_to_storage = myDatasetsPath #'D:\\Faina\\StoragePath\\Datasets'

    ds = 'A4'
    args.data_path = os.path.join(args.path_to_storage, 'Counting Datasets', 'CVPPP2017_LCC_training', 'training', ds)


    args.test_csv_leaf_number_file = os.path.join(args.data_path, 'val', ds + '_Val.csv')
    args.test_csv_leaf_location_file = os.path.join(args.data_path, 'val', ds + '_Val_leaf_location.csv')

    # The path to the model+weights file
    dir_name = 'stas comp_op 20'  #LCC_torch pre processing #LCC_retinanet_weights
    args.model_path = 'legonet_final.pt' #os.path.join('D:\\PyCharmProjects\\M', dir_name, 'model_final.pt')

    # True- if the pt file contains the model, not only the weigths
    args.pt_with_model = True

    args.option = 20 #" "
    args.ann_type = 'count'
    args.calc_det_performance = False # Doesn't work with True

    args.pre_process = 'keras_like'

    dataset_val = csv_LCCDataset(
        args.test_csv_leaf_number_file,
        args.test_csv_leaf_location_file,
        option=args.option,
        pre_process='keras_like',
        ann_type='count',
        transform=transforms.Compose([Normalizer(pre_process=args.pre_process, ann_type='count', option=args.option),
                                      Resizer(ann_type='count', option=args.option, min_side=800, max_side=1333)]))

    use_gpu = True

    if args.pt_with_model:
        legonet = torch.load(args.model_path)
        if use_gpu:
            legonet = legonet.cuda()

    else:
        # If using only a weights file (that doesn't contain the model) - Create the model
        ######################################################################################################
        ### Todo - fix it
        legonet = model.resnet50(args, num_classes=dataset_val.num_classes(), pretrained=True)
        #######################################################################################################
        if use_gpu:
            legonet = legonet.cuda()

        # load the weights
        legonet.load_state_dict(torch.load(args.model_path))
        legonet.module.freeze_bn()


    legonet = torch.nn.DataParallel(legonet).cuda()

    legonet.training = False
    legonet.eval()


    counting_eval.eval(dataset_val, legonet, args)


if __name__ == '__main__':
    main()
