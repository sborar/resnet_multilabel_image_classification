import glob
import gzip
import numpy as np
import os
import pandas as pd
from os.path import join, isdir
from PIL import Image
import pickle
import torch
import sklearn
from sklearn.model_selection import train_test_split

# def train_test_split_with_sampling(df):

# use the meta name to put things in a certain folder
base = "/azure-ml/mvinterns/deepmind-headneck-0/ct/"

image_dataname = "image*"

raw_data_folder = "raw_dataset"

list_parts = ['Brain', 'Bone_Mandible', 'SpinalCanal', 'Glnd_Lacrimal_L', 'Lung_R', 'Glnd_Submand_R', 'Glnd_Lacrimal_R',
              'Cochlea_L', 'OpticNrv_cnv_R', 'Lens_R', 'SpinalCord', 'Parotid_R', 'Glnd_Submand_L', 'Brainstem',
              'OpticNrv_cnv_L', 'Cochlea_R', 'Eye_R', 'Lens_L', 'Lung_L','Eye_L', 'Parotid_L']

# get all files with pattern
image_files = glob.glob(join(base, image_dataname))

# for each file
final_data = pd.DataFrame(
    columns=['img'] + list_parts)
count = 0

for image_file in image_files:
    img_data = pd.DataFrame(
        columns=['img'] + list_parts)
    try:
        pid_scanid = image_file.split('/')[-1][6:-4]

        print(pid_scanid)
        pid = pid_scanid.split('.')[0]
        scanid = pid_scanid.split('-')[-1].split('_')[0]

        meta_dataname = join(base, "meta_" + pid_scanid + ".pkz")
        image = np.load(image_file, allow_pickle=True)["arr_0"]
        meta_data = pickle.load(gzip.open(meta_dataname, 'rb'))

        masksname = join(base, "masks_" + pid_scanid + "/")
        masks = []

        for filename in os.listdir(masksname):
            f = join(masksname, filename)

            # checking if it is a file
            mask = np.load(gzip.open(f, 'rb'), allow_pickle=True)
            masks.append(mask)
    except Exception as e:
        print(e)
        continue

    if not masks:
        continue

    shape = meta_data['shape']  # scan shape
    spacing = (meta_data['ND_SliceSpacing'], meta_data['PixelSpacing'][1], meta_data['PixelSpacing'][0])

    for mask_data in masks:
        count = 0
        shape = mask_data['shape']  # shape of scan
        bbox = mask_data['bbox']
        cropped_mask = mask_data['roi']
        body_part_name = mask_data['name']

        if body_part_name in list_parts:
            print('Saving ' + body_part_name)
            x = image.shape[0]
            body_part_folder = join(raw_data_folder, body_part_name)
            pid_scanid_folder = join(body_part_folder, "img", pid, scanid)
            if not isdir(pid_scanid_folder):
                os.makedirs(pid_scanid_folder)

            if cropped_mask is None:
                print('roi is none')
                for i in range(x):
                    img_name = join(body_part_folder, 'img', pid, scanid, pid_scanid + "_" + str(i) + ".png")
                    Image.fromarray(np.uint8(image[i, :, :])).convert('RGB').save(img_name)
                    img_data.loc[count, 'img'] = img_name
                    img_data.loc[count, body_part_name] = -1
                    count += 1
                continue
            else:

                mask = np.zeros(shape, dtype=np.bool)
                try:
                    b = [bbox[i] for i in [0, 3, 1, 4, 2, 5]]  # get it in (z_min, z_max, y_min, y_max, x_min, x_max)
                    mask[b[0]: b[1], b[2]: b[3], b[4]: b[5]] = cropped_mask
                    z_min = bbox[0]
                    z_max = bbox[3]

                    mask_rgb = (mask[:, :, :] * 255).astype(np.uint8)

                    for i in range(x):
                        img_name = join(body_part_folder, 'img', pid, scanid, pid_scanid + "_" + str(i) + ".png")
                        Image.fromarray(np.uint8(image[i, :, :])).convert('RGB').save(
                            img_name)
                        img_data.loc[count, 'img'] = img_name
                        if i < z_min or i > z_max:
                            img_data.loc[count, body_part_name] = -1
                        else:
                            img_data.loc[count, body_part_name] = 1
                        count += 1
                except Exception as e:
                    print(e)
                    continue
    final_data = final_data.append(img_data)
    print()

final_data = final_data.fillna(0)

final_data.to_csv('img_data.csv', index=None)
# img_data = pd.read_csv('img_data.csv')
# print(img_data.columns)
labels = final_data.drop(['img'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(final_data['img'], labels, test_size=0.20,
                                                      random_state=42)

train_data = pd.DataFrame()
train_data['img'] = X_train
train_data.loc[:, labels.columns] = y_train

#
# X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.50,
#                                                     random_state=42, stratify=y_test)
#
# test_data = pd.DataFrame()
# test_data['img'] = X_test
# test_data.loc[:, labels.columns] = y_test

valid_data = pd.DataFrame()
valid_data['img'] = X_valid
valid_data.loc[:, labels.columns] = y_valid

if not os.path.exists('data'):
    os.makedirs('data')
# test_data.to_csv('data/test_data.csv', index=None)
train_data.to_csv('data/train_data.csv', index=None)
valid_data.to_csv('data/valid_data.csv', index=None)
