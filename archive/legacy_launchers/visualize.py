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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np

import time
import argparse

import cv2
import config

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from legonet.myDataloader import KCSVDataset, CocoDataset, kcsv_collater, Resizer, AspectRatioBasedSampler, UnNormalizer, Normalizer



assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

config.General.network_type = config.NetworkType.detection

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	# parser.dataset = "coco"
	# parser.coco_path = "C:\\Users\\stas\\Desktop\\tomato_fruit_12_3_18"
	# parser.model = "legonet\\coco_legonet_49.pt"

	parser.dataset = "kcsv"

	parser.kcsv_classes = "C:\\Users\\Aragorn\\Desktop\\Datasets\\Faina_datasets\\task_122_banana_bunch_detection\\classes.kcsv"
	parser.kcsv_train 	= "C:\\Users\\Aragorn\\Desktop\\Datasets\\Faina_datasets\\task_122_banana_bunch_detection\\train\\train.kcsv"
	parser.kcsv_val 	= "C:\\Users\\Aragorn\\Desktop\\Datasets\\Faina_datasets\\task_122_banana_bunch_detection\\val\\val.kcsv"
	parser.kcsv_test 	= "C:\\Users\\Aragorn\\Desktop\\Datasets\\Faina_datasets\\task_122_banana_bunch_detection\\test\\test.kcsv"

	parser.model = "C:\\Users\\Aragorn\\Desktop\\Experiments\\banana_bunch_segmentation\\legonet_epoch=25_new.pt"

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'kcsv':
		# dataset_val = KCSVDataset(train_file=parser.kcsv_train, class_list=parser.kcsv_classes, transform=transforms.Compose([Normalizer(), Resizer(ann_type="bbox")]))
		dataset_val = KCSVDataset(input_file=parser.kcsv_test, class_list=parser.kcsv_classes,
                                  pre_process = 'torch_like',
                                  transform=transforms.Compose([Normalizer(pre_process = 'torch_like'), Resizer(min_side=800, max_side=1333)]),
                                  lean_version= 'version_3')
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False, do_shuffle=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=kcsv_collater, batch_sampler=sampler)

	legonet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		legonet = legonet.cuda()

	legonet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):
		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			st = time.time()
			detection_outputs = legonet(data['img'].cuda().float())
			scores, classification, transformed_anchors = detection_outputs
			print('Elapsed time: {}'.format(time.time()-st))

			idxs = np.where(scores.cpu()>0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			# draw predictions
			for j in range(idxs[0].shape[0]):
				ann = transformed_anchors[idxs[0][j], :]
				x1 = int(ann[0])
				y1 = int(ann[1])
				x2 = int(ann[2])
				y2 = int(ann[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				#draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				#print(label_name)

			# draw GT
			annots = data["bbox_annot"].numpy()[0]
			for ann in annots:
				x1 = int(ann[0])
				y1 = int(ann[1])
				x2 = int(ann[2])
				y2 = int(ann[3])
				label_name = dataset_val.labels[int(ann[4])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
				#print(label_name)

			cv2.imshow('img', img)
			cv2.waitKey(0)
