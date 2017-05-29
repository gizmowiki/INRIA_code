import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET


def visualize(base_path):
    for xmlfiles in os.listdir(os.path.join(base_path, 'Annotations')):
        if xmlfiles.endswith('.xml'):
            base_name = xmlfiles.split('.')[0]
            tree = ET.parse(os.path.join(base_path, 'Annotations', xmlfiles))
            root = tree.getroot()
            img = cv2.imread(os.path.join(base_path, 'JPEGImages', base_name + ".jpeg"), 1)
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []

            for xmin in root.iter('xmin'):
                xmins.append(int(float(xmin.text)))
            for xmax in root.iter('xmax'):
                xmaxs.append(int(float(xmax.text)))
            for ymax in root.iter('ymax'):
                ymaxs.append(int(float(ymax.text)))
            for ymin in root.iter('ymin'):
                ymins.append(int(float(ymin.text)))

            for i in range(len(xmins)):
                cv2.rectangle(img, (xmins[i], ymins[i]), (xmaxs[i], ymaxs[i]), (0, 0, 255), thickness=2)

            if os.path.exists(os.path.join(base_path, 'negative_truth_updated', base_name + '.txt')):
                for lines in open(os.path.join(base_path, 'negative_truth_updated', base_name + '.txt')):
                    lines = lines.strip()
                    lines = lines.split('\t')
                    negbox = [int(x) for x in lines]
                    cv2.rectangle(img, (negbox[0], negbox[1]), (negbox[2], negbox[3]), (0, 255, 0), thickness=2)

            cv2.imshow("Hollywood heads", img)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break


def print_fn(data):
    print (str(data)),


def generate_data(base_path):
    for xmlfiles in os.listdir(os.path.join(base_path, 'Annotations')):
        if xmlfiles.endswith('.xml'):
            base_name = xmlfiles.split('.')[0]
            if not os.path.exists(os.path.join(base_path, 'crop', 'positive', base_name)):
                os.makedirs(os.path.join(base_path, 'crop', 'positive', base_name))
            if not os.path.exists(os.path.join(base_path, 'crop', 'negative', base_name)):
                os.makedirs(os.path.join(base_path, 'crop', 'negative', base_name))
            tree = ET.parse(os.path.join(base_path, 'Annotations', xmlfiles))
            root = tree.getroot()
            img = cv2.imread(os.path.join(base_path, 'JPEGImages', base_name + ".jpeg"), 1)
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []

            for xmin in root.iter('xmin'):
                xmins.append(int(float(xmin.text)))
            for xmax in root.iter('xmax'):
                xmaxs.append(int(float(xmax.text)))
            for ymax in root.iter('ymax'):
                ymaxs.append(int(float(ymax.text)))
            for ymin in root.iter('ymin'):
                ymins.append(int(float(ymin.text)))
	    
	    xmins=list((np.asarray(xmins)).clip(min=0))
            ymins=list((np.asarray(ymins)).clip(min=0))
            xmaxs=list((np.asarray(xmaxs)).clip(min=0))
            ymaxs=list((np.asarray(ymaxs)).clip(min=0))
	    
            for i in range(len(xmins)):
                img_cropped_nsize = img[ymins[i]:ymaxs[i], xmins[i]:xmaxs[i]]
                img_cropped = cv2.resize(img_cropped_nsize, (256, 256), interpolation=cv2.INTER_CUBIC)
                img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                imgname = os.path.join(base_path, 'crop', 'positive', base_name, "{0:03d}.jpg".format(i))
                cv2.imwrite(imgname, img_cropped)
                print_fn(i)

            if os.path.exists(os.path.join(base_path, 'negative_truth_updated', base_name + '.txt')):
                cnt = 0
                for lines in open(os.path.join(base_path, 'negative_truth_updated', base_name + '.txt')):
                    lines = lines.strip()
                    lines = lines.split('\t')
                    negbox = [int(x) for x in lines]
                    img_cropped_nsize = img[negbox[1]:negbox[3], negbox[0]:negbox[2]]
                    img_cropped = cv2.resize(img_cropped_nsize, (256, 256), interpolation=cv2.INTER_CUBIC)
                    img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                    imgname = os.path.join(base_path, 'crop', 'negative', base_name, "{0:03d}.jpg".format(cnt))
                    cv2.imwrite(imgname, img_cropped)
                    print_fn(cnt)
                    cnt += 1

            print ("Completed! base_path %s image name %s" % (base_path, base_name))

# visualize('/user/rpandey/home/inria/code/cnnheaddetection/cnn_head_detection/data/HollywoodHeads/')
generate_data('/data/stars/user/rpandey/head_dataset/')
