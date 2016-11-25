
# coding: utf-8

# In[24]:

import cv2, os, shutil
import xml.etree.ElementTree as ET


# In[2]:

ann_DIR = "/opt/py-faster-rcnn/data/VOCdevkit2007/VOC2007.cd24/Annotations"
img_DIR = "/opt/py-faster-rcnn/data/VOCdevkit2007/VOC2007.cd24/JPEGImages"
trainval_file = "/opt/py-faster-rcnn/data/VOCdevkit2007/VOC2007.cd24/ImageSets/Main/trainval.txt"
test_file = "/opt/py-faster-rcnn/data/VOCdevkit2007/VOC2007.cd24/ImageSets/Main/test.txt"


# In[42]:

def readXmlAnno(im_fn, ann_DIR):
    anno_pn = os.path.join(ann_DIR, im_fn+'.xml')
    #print 'On annotation: {}'.format(anno_pn)
    tree = ET.parse(anno_pn)
    root = tree.getroot()
    
    p_anno = {}
    size = root.find('size')
    d_size = {"width": size.find('width').text, 
              "height": size.find('height').text,
              "depth": size.find('depth').text
             }        
    p_anno['size'] = d_size
    
    l_obj = []
    for obj in root.findall('object'):
        d_obj = {"name": obj.find('name').text, "truncated": '0.0', "difficult": '0.0', "occluded":'0.0',
                 "xmin": float(obj.find('bndbox').find('xmin').text),
                 "ymin": float(obj.find('bndbox').find('ymin').text),
                 "xmax": float(obj.find('bndbox').find('xmax').text),
                 "ymax": float(obj.find('bndbox').find('ymax').text),
                }
        l_obj.append(d_obj)
    
    p_anno['l_obj'] = l_obj
        
    if len(l_obj) > 0:       
        return p_anno
    else:
        return None    
    
def convertToKitti(p_anno):
    l_annos = []
    for obj in p_anno['l_obj']:
        k_anno = {}
        k_anno["type"] = obj["name"]
        k_anno["truncated"] = obj["truncated"]
        k_anno["occluded"] = obj["occluded"]
        k_anno["alpha"] = '0.0'
        k_anno["bbox"] = "{:.1f} {:.1f} {:.1f} {:.1f}".format(obj["xmin"],obj["ymin"],obj["xmax"],obj["ymax"])
        k_anno["dimensions"] = "{:.1f} {:.1f} {:.1f}".format(0,0,0)
        k_anno["location"] = "{:.1f} {:.1f} {:.1f}".format(0,0,0)
        k_anno["rotation_y"] = '0.0'
        l_annos.append(k_anno)
        
    return l_annos
    


# ### Create our own kitti dataset

# In[23]:

k_train_img_DIR = "/opt/data/kitti_cd/train/images"
k_train_lab_DIR = "/opt/data/kitti_cd/train/labels"
k_val_img_DIR = "/opt/data/kitti_cd/val/images"
k_val_lab_DIR = "/opt/data/kitti_cd/val/labels"


# In[47]:

# Processing train data
#'''
with open(trainval_file) as in_f:
    for im_fn in in_f:
        #print 'Processing img: {}'.format(im_fn)
        im_fn = im_fn.split('\n')[0].split('\r')[0]
        p_anno = readXmlAnno(im_fn, ann_DIR)
        if p_anno != None:
            l_annos = convertToKitti(p_anno)
            
            k_anno_file = os.path.join(k_train_lab_DIR,im_fn+".txt")
            with open(k_anno_file, 'w') as out_f:
                for k_anno in l_annos:
                    #print k_anno
                    to_write = ""
                    to_write += str(k_anno['type'])+' '
                    to_write += str(k_anno['truncated'])+' '
                    to_write += str(k_anno['occluded'])+' '
                    to_write += str(k_anno['alpha'])+' '
                    to_write += str(k_anno['bbox'])+' '
                    to_write += str(k_anno['dimensions'])+' '
                    to_write += str(k_anno['location'])+' '
                    to_write += str(k_anno['rotation_y'])
                    #print to_write
                    out_f.write(to_write+'\n')
                out_f.close()
            
            # copy im file
            from_file = os.path.join(img_DIR,im_fn+".jpg")
            to_file = os.path.join(k_train_img_DIR,im_fn+".jpg")
            shutil.copy2(from_file, to_file)
            #break
    in_f.close()
#'''

# Processing test data
with open(test_file) as in_f:
    for im_fn in in_f:
        #print 'Processing img: {}'.format(im_fn)
        im_fn = im_fn.split('\n')[0].split('\r')[0]
        p_anno = readXmlAnno(im_fn, ann_DIR)
        if p_anno != None:
            l_annos = convertToKitti(p_anno)
            
            k_anno_file = os.path.join(k_val_lab_DIR,im_fn+".txt")
            with open(k_anno_file, 'w') as out_f:
                for k_anno in l_annos:
                    #print k_anno
                    to_write = ""
                    to_write += str(k_anno['type'])+' '
                    to_write += str(k_anno['truncated'])+' '
                    to_write += str(k_anno['occluded'])+' '
                    to_write += str(k_anno['alpha'])+' '
                    to_write += str(k_anno['bbox'])+' '
                    to_write += str(k_anno['dimensions'])+' '
                    to_write += str(k_anno['location'])+' '
                    to_write += str(k_anno['rotation_y'])
                    #print to_write
                    out_f.write(to_write+'\n')
                out_f.close()
            
            # copy im file
            from_file = os.path.join(img_DIR,im_fn+".jpg")
            to_file = os.path.join(k_val_img_DIR,im_fn+".jpg")
            shutil.copy2(from_file, to_file)
            #break
    in_f.close()


# In[ ]:



