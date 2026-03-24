import os
import uuid
import json
import cv2
import numpy as np
import sys
from PIL import Image


class RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json_data
        self.uuid = str(uuid.uuid4())


# task:229,241/2/3 exp:142
def get_data_by_taskID(task_id):
    res = p.get_images_by_task_id(task_id)
    if len(res) == 0:
        print('There are no images for this task!')
    else:
        saved_im_dir = "Data/tasks/" + str(task_id)
        if not os.path.exists(saved_im_dir):
            os.makedirs(saved_im_dir)
        if not os.path.exists(saved_im_dir + '/images'):
            os.makedirs(saved_im_dir + '/images')
        if not os.path.exists(saved_im_dir + '/annotations'):
            os.makedirs(saved_im_dir + '/annotations')
        for i in range(0, len(res)):
            image_data = res[i]
            image_id = image_data['image_id']
            print('working on image {}'.format(image_id))
            uri = image_data['image_uri']
            im = Image.open(uri)
            im_copy = Image.open(uri)
            im_copy = im_copy.resize((848, 480))
            # im_copy.show()
            if not uri.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            im.save(saved_im_dir + '/images/' + str(uri).split('/')[-1])
            image_meta_data = p.get_image_data(str(image_id))
            if len(image_meta_data) == 0:
                print('Image {} has no annotation file'.format(image_id))
                continue
            if isinstance(image_meta_data['annotator'], np.basestring):
                print('Image {} has no annotation file'.format(image_id))
                continue
            annotators = list(image_meta_data['annotator'].keys())
            if len(annotators) == 0:
                print('Image {} has no annotation file'.format(image_id))
            for annotator in annotators:
                try:
                    json_content = image_meta_data['annotator'][annotator]['json_content']
                except:
                    print('Image {} has no annotation file'.format(image_id))
                    continue
                with open(saved_im_dir + '/annotations/{}.json'.format(uri.split('/')[-1].split('.')[0]), 'w') as f:
                    json.dump(json_content, f)
    print('done downlading data of task {}'.format(task_id))


def get_data_by_experimentID(experiment_id):
    res = p.get_images_by_experiment_id(experiment_id)
    if len(res) == 0:
        print('There are no images for this experiment!')
    else:
        saved_im_dir = "Data/experiments/" + str(experiment_id)
        if not os.path.exists(saved_im_dir):
            os.makedirs(saved_im_dir)
        if not os.path.exists(saved_im_dir + '/images'):
            os.makedirs(saved_im_dir + '/images')
        if not os.path.exists(saved_im_dir + '/annotations'):
            os.makedirs(saved_im_dir + '/annotations')
        for i in range(0, len(res)):
            image_data = res[i]
            image_id = image_data['image_id']
            print('working on image {}'.format(image_id))
            uri = image_data['image_uri']
            im = Image.open(uri)
            im_copy = Image.open(uri)
            im_copy = im_copy.resize((848, 480))
            # im_copy.show()
            if not uri.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            im.save(saved_im_dir + '/images/' + str(uri).split('/')[-1])
            image_meta_data = p.get_image_data(str(image_id))
            if len(image_meta_data) == 0:
                print('Image {} has no annotation file'.format(image_id))
                continue
            if isinstance(image_meta_data['annotator'], np.basestring):
                print('Image {} has no annotation file'.format(image_id))
                continue
            annotators = list(image_meta_data['annotator'].keys())
            if (len(annotators) == 0):
                print('Image {} has no annotation file'.format(image_id))
            for annotator in annotators:
                try:
                    json_content = image_meta_data['annotator'][annotator]['json_content']
                except:
                    print('Image {} has no annotation file'.format(image_id))
                    continue
                with open(saved_im_dir + '/annotations/{}.json'.format(uri.split('/')[-1].split('.')[0]), 'w') as f:
                    json.dump(json_content, f)
    print('done downlading data of experiment {}'.format(experiment_id))

def get_images_by_organ(organ):

    res = p.get_images_by_organ('stem','0')
    print (len(res))
    if len(res) == 0:
        print('There are no images for this organ!')
    else:
        saved_im_dir = "Data/organs/" + str(organ)
        if not os.path.exists(saved_im_dir):
            os.makedirs(saved_im_dir)
        if not os.path.exists(saved_im_dir + '/images'):
            os.makedirs(saved_im_dir + '/images')
        if not os.path.exists(saved_im_dir + '/annotations'):
            os.makedirs(saved_im_dir + '/annotations')
        for i in range(0, len(res)):
            image_data = res[i]
            image_id = image_data['image_id']
            print('working on image {}'.format(image_id))
            uri = image_data['image_uri']
            im = Image.open(uri)
            im_copy = Image.open(uri)
            im_copy = im_copy.resize((848, 480))
            # im_copy.show()
            if not uri.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            im.save(saved_im_dir + '/images/' + str(uri).split('/')[-1])
            image_meta_data = p.get_image_data(str(image_id))
            if len(image_meta_data) == 0:
                print('Image {} has no annotation file'.format(image_id))
                continue
            if isinstance(image_meta_data['annotator'], np.basestring):
                print('Image {} has no annotation file'.format(image_id))
                continue
            annotators = list(image_meta_data['annotator'].keys())
            if (len(annotators) == 0):
                print('Image {} has no annotation file'.format(image_id))
            for annotator in annotators:
                try:
                    json_content = image_meta_data['annotator'][annotator]['json_content']
                except:
                    print('Image {} has no annotation file'.format(image_id))
                    continue
                with open(saved_im_dir + '/annotations/{}.json'.format(uri.split('/')[-1].split('.')[0]), 'w') as f:
                    json.dump(json_content, f)
    print('done downlading data of organ {}'.format(organ))


# res = p.get_imaging_tasks()

# res=p.get_image_data_by_inner_key('989','annotator.danielk')[0]['annotator.danielk']['json_content']


if __name__ == '__main__':
    func = str(sys.argv[1])
    idd = str(sys.argv[2])

    func = "experiment"
    idd = 119

    #id =469
    #func = 'experiment'
    assert func in ['task', 'experiment','organ'], "can retrieve images ony from task or experiment"
    p = Phenomics_API()
    p.login('guyf', 'Amtgo192376')
    p.auth = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1NDA5ODMxNzksImlhdCI6MTU0MDk4MTM3OSwibmJmIjoxNTQwOTgxMzc5LCJpZGVudGl0eSI6eyJpZCI6ImFkYXJ2IiwidXNlcl9uYW1lIjoiYWRhcnYiLCJmaXJzdF9uYW1lIjoiQWRhciIsImxhc3RfbmFtZSI6IlZpdCIsImRpc3BsYXlfbmFtZSI6IkFkYXIgVml0IiwiZ3JvdXBzIjpbInNoYW5pIiwiaXBhdXNlcnMiXX19.aGT6hqdpeshzN483GYcRv5hhMPMF8kw357HUEp9VXWI"
    if func == 'task':
        get_data_by_taskID(str(idd))
    if func == 'experiment':
        get_data_by_experimentID(str(idd))
    if func == 'organ':
        get_images_by_organ(str(idd))