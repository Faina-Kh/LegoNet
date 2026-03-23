import os
import sys

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIR = CURRENT_DIR
while not (os.path.exists(os.path.join(SEARCH_DIR, "config.py")) and os.path.isdir(os.path.join(SEARCH_DIR, "legonet"))):
    parent = os.path.dirname(SEARCH_DIR)
    if parent == SEARCH_DIR:
        break
    SEARCH_DIR = parent
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torchvision
import time

import copy
import pdb
import time
import argparse

import config

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from legonet.dataloader import CocoDataset, kcsv_collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer

from PIL import Image, ImageDraw, ImageFont

from legonet.kcsv_dataloader import KCSVDataset


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

config.General.network_type = config.NetworkType.detection
config.Detection.NMS_THRESHOLD = 0.3

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	# parser.dataset = "coco"
	# parser.coco_path = "C:\\Users\\stas\\Desktop\\tomato_fruit_12_3_18"
	# parser.model = "legonet\\coco_legonet_49.pt"

	parser.dataset = "kcsv"

	parser.kcsv_classes = "D:\\Faina\\StoragePath\\Datasets\\KK_datasets\\131_wheat_spikes_and_spikelets\\classes.kcsv"
	parser.kcsv_train = "D:\\Faina\\StoragePath\\Datasets\\KK_datasets\\131_wheat_spikes_and_spikelets\\train\\train_MS5.kcsv"
	parser.kcsv_val = "D:\\Faina\\StoragePath\\Datasets\\KK_datasets\\131_wheat_spikes_and_spikelets\\val\\val_MS5.kcsv"
	parser.kcsv_test = "D:\\Faina\\StoragePath\\Datasets\\KK_datasets\\131_wheat_spikes_and_spikelets\\test\\test_MS5.kcsv"

	parser.model = "wheatnewannots\\bestsofar\\legonet_epoch=150.pt"

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'kcsv':
		# dataset_val = KCSVDataset(train_file=parser.kcsv_train, class_list=parser.kcsv_classes, transform=transforms.Compose([Normalizer(), Resizer(ann_type="bbox")]))
		dataset_val = KCSVDataset(train_file=parser.kcsv_test, class_list=parser.kcsv_classes,
								  pre_process = 'torch_like',
								  transform=transforms.Compose([Normalizer(pre_process = 'torch_like'), Resizer(min_side=800, max_side=1333)]),
								  lean_version= 'version_3')
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False, do_shuffle=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=kcsv_collater, batch_sampler=sampler, shuffle=False)

	legonet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		legonet = legonet.cuda()

	legonet.eval()
	legonet.network_type = "detection"

	unnormalize = UnNormalizer()

	font = ImageFont.truetype('arial.ttf', 14)
	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			st = time.time()
			detection_outputs = legonet([data['img'].cuda().float(), [data['bbox_annot'], None], None, False])
			scores, classification, transformed_anchors = detection_outputs
			print('Elapsed time: {}'.format(time.time()-st))

			idxs = np.where(scores.cpu() > 0.05)
			img_array = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img_array[img_array < 0] = 0
			img_array[img_array > 255] = 255

			img_array = np.transpose(img_array, (1, 2, 0))

			img = Image.fromarray(np.uint8(img_array))

			draw = ImageDraw.Draw(img)

			# img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			# draw predictions
			for j in range(idxs[0].shape[0]):
				ann = transformed_anchors[idxs[0][j], :]
				x1 = int(ann[0])
				y1 = int(ann[1])
				x2 = int(ann[2])
				y2 = int(ann[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				score = scores[idxs[0][j]]

				draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=config.DrawProperties.LINE_WIDTH)
				draw.text((x1, y2-20), "score = {:.3f}".format(score.item()), font=font)



			# draw GT
			annots = data["bbox_annot"].numpy()[0]
			for ann in annots:
				x1 = int(ann[0])
				y1 = int(ann[1])
				x2 = int(ann[2])
				y2 = int(ann[3])
				label_name = dataset_val.labels[int(ann[4])]

				draw.rectangle(((x1, y1), (x2, y2)), outline="green", width=config.DrawProperties.LINE_WIDTH)
				draw.text((x1, y1+15), label_name, font=font)

			img.show()
