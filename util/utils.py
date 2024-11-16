
import os.path
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from configration import CAN_DATA_DIR,BASE_DIR

file_paths = [
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"),
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"),
    os.path.join(CAN_DATA_DIR, "MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"),
]
pickle_dict = {
    're-primary': os.path.join(CAN_DATA_DIR, "pickle/re-primary.pickle"),
    "re-standarlization": os.path.join(CAN_DATA_DIR, "pickle/re-standarlization.pickle")
}


def read_data(file):
    print("\nreading data {}\n".format(file))
    row_data = pd.read_csv(file, header=None, low_memory=False)
    return row_data.drop([0])


def clean_data(data):
    dropList = data[(data[14] == 'Nan') | (data[15] == "Infinity")].index.tolist()
    return data.drop(dropList)


def summary_data(data):
    last_colum_index = data.columns[-1]
    return data[last_colum_index].value_counts()


def read_mycsv(file_paths):
    """
    返回所有csv的dataframe;清理了Nan和Infinity;
    :param file_paths: list
    :return: result: dataframe;
    """
    result = pd.DataFrame([])
    for file_path in file_paths:
        result_tmp = read_data(file_path)
        result_tmp = clean_data(result_tmp)
        result = pd.concat([result, result_tmp])
    print(summary_data(result))
    return result


def standarlization(data):
    """
    对data进行归一化，用于后面转图片
    :param data: dataframe
    :return: dataframe
    """
    data_label = data[data.columns[-1]]
    data = data.iloc[:, :-2]
    # 强制类型转换
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(0)
    # 归一化
    data_scal = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 255 if np.max(x) > np.min(x) else 0
    data = data.apply(data_scal)

    return pd.concat([data, data_label], axis=1)


def convert2img(data,img_shape=[6,6]):
    """
    把一条数据转为img
    :param data: dataframe(一条)
    :return: img
    """
    img_matrix = data.values.reshape(img_shape[0], img_shape[1]).astype(np.uint8)
    return Image.fromarray(img_matrix)


def save_img(img, file_name):
    """
    存储img
    :param img: PIL.Image
    :return: bool
    """
    file_path = os.path.dirname(file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    try:
        img.save(file_name)
        return True
    except FileNotFoundError:
        print('image{}保存失败'.format(file_name))
        return False


def read_pickle(pickle_path: str):
    """
    读取pickle文件,没有则返回空
    :param pickle_path:
    :return:
    """
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("读取pickle文件{}出错".format(pickle_path))
            return None
    return None


def creat_pickle(pickle_path: str, data):
    """

    :param pickle_path: str;
    :param data: any;要存储的数据
    :return: bool
    """
    dir_path = os.path.dirname(pickle_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except FileNotFoundError:
        print("写入pickle文件{}时出错".format(pickle_path))
        return False


def sample_data(data, rate):
    """
    数据采样，后面会补充如滑动窗口采样的功能
    :param data: dataframe
    :param rate: float
    :return: dataframe
    """
    return data.sample(frac=rate)


def try_save_img(re_standarlize, img_shape=[6,6],sample_rate=0.0001,out_path = r"D:\CodeSpace\Python\NetLearn\CNN\data\image"):
    """
    采样并存储一些img
    :param re_standarlize:
    :return:
    """
    data_sample = sample_data(re_standarlize, sample_rate)
    root_dir = out_path
    for index, row in data_sample.iterrows():
        label = row.iloc[-1]
        file_path = os.path.join(root_dir, label)
        file_name = "{}_{}.png".format(label, index)
        img = convert2img(row[:64],img_shape)
        is_write = save_img(img, os.path.join(file_path, file_name))
        if is_write:
            if os.path.exists(file_path):
                with open(os.path.join(file_path, "image_name.txt"), 'a+') as f:
                    f.write("{}\n".format(file_name))


def try_read_img(root_path):
    labels = []
    imgs = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        print('walking dir:\t{}'.format(dirpath))
        for filename in filenames:
            if filename.split('.')[-1] == 'txt':
                continue
            else:
                labels.append(filename.split('_')[0])
                # 读取图片并转为np
                img_tmp = Image.open(os.path.join(dirpath, filename))
                img_tmp = np.array(img_tmp)
                imgs.append(img_tmp)
    print("imgs :\ntype:{}\ndata:{}\n".format(type(imgs[0]), imgs[:3]))
    print("labels :\ntype:{}\ndata:{}\n".format(type(labels[0]), labels[:3]))

    return imgs, labels

def create_index_list(out_path,session_list,walk_dir=r"D:\CodeSpace\Python\NetLearn\CNN\data\image"):
    """
    遍历目录，并生成index_list文件夹
    Args:
        out_path: index_list输出路径（为目录）
        walk_dir: 要遍历的目录（特殊定制）
        session_list: 每个session对应的类别样例：{
            1 : ['Begin','DDos']
            2 : ['PortScan']}

    Returns:

    """
    # 创建目录
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #遍历,每个子文件夹存一个dict项
    root_path = walk_dir
    dirs = []
    for dir,dirnames,filenames in os.walk(root_path):
        dirs = dirnames
        break
    index_dict = dict()
    #先组成一个dict，后面再分类，好改代码
    for dirname in dirs:
        for nowpath,_,filenames in os.walk(os.path.join(root_path,dirname)):
            for filename in filenames:
                if 'txt' in filename:
                    with open(os.path.join(nowpath,filename),'r') as f:
                        dict_list = f.readlines()
                        dict_list = [os.path.join(nowpath,x) for x in dict_list]
                        index_dict.update({dirname:dict_list})
                    break
    #分类
    for key,values in session_list.items():
        file_name = "session_{}.txt".format(key)
        with open(os.path.join(out_path,file_name),'a+') as f:
            for value in values:
                f.writelines(index_dict[value])
    pass

def CEC_save_img(re_standarlize,sample_rate=0.0001,out_index_path='../data/index_list/cicids2017'):
    """
    CEC的生成图片，同时配套生成index_list
    Args:
        re_standarlize: 标准化后的re
        sample_rate: 采样频率
        out_index_path: 输出index_list的路径

    Returns:

    """
    try_save_img(re_standarlize,sample_rate,os.path.join(CAN_DATA_DIR,'image'))
    session_list = dict({
        1: ['BENIGN', 'DDoS'],
        2: ['DoS GoldenEye']
    })
    create_index_list(out_index_path,session_list,os.path.join(CAN_DATA_DIR,'image'))

def CEC_read_img(index_path):
    """
    根据index_path读取图片和Label
    Args:
        index_path: a txt which includes the imgs' path

    Returns: (imgs,labels)
            where imgs looks like : [np.array,np.array...]
            and labels looks like : [str,str...]

    """
    data_path = os.path.expanduser(index_path)
    img_paths = []
    with open(data_path,'r') as f:
        img_paths = f.readlines()
    #处理图片
    labels = []
    imgs = []
    for img_path in img_paths:
        img_tmp = Image.open(img_path[:-1])
        imgs.append(np.array(img_tmp))
        labels.append(process_img_path(img_path))
    return imgs,labels


def process_img_path(img_path):
    """

    Args:
        img_path: like : D:\\CodeSpace\\Python\\NetLearn\\CNN\\data\\image\\BENIGN\\BENIGN_61943.png

    Returns: label of the img
            like : BENIGN
    """
    if '\\' in img_path:
        return img_path.split('\\')[-2]
    else:
        return img_path.split('/')[-2]

if __name__ == '__main__':
    # result = read_mycsv(file_paths)
    # if read_pickle(pickle_dict['re-primary']) is None:
    #     creat_pickle(pickle_dict['re-primary'],result)
    '''读取result到saveimage测试'''
    # result = read_pickle(pickle_dict['re-primary'])
    # print(summary_data(result))
    # if read_pickle(pickle_dict['re-standarlization']) is None:
    #     creat_pickle(pickle_dict['re-standarlization'],standarlization(result))
    # result_standarlize = read_pickle(pickle_dict['re-standarlization'])
    # print(result_standarlize.head(15))
    # print(summary_data(result_standarlize))
    # try_save_img(result_standarlize)
    '''尝试读取图片测试'''
    # imgs,labels = try_read_img(os.path.join(CAN_DATA_DIR,"image"))
    # print(len(imgs))
    # le = LabelEncoder()
    # print(le.fit_transform(labels))
    '''调试读取txt'''
    # txt_path = os.path.join(CAN_DATA_DIR,'image/DDos/image_name.txt')
    # with open(txt_path,'r') as f:
    #     print(f.readlines()) # result : DDoS_145897.png\n
    '''list的复制配合os.walk'''
    # rootpath = os.getcwd()[:-5]
    # for d,di,fi in os.walk(rootpath):
    #     #list[:]= 是在原地修改列表，而 list= 是创建一个新的列表对象并重新赋值给变量
    #     ##这里需要给原列表修改
    #     print(f'walking d : {d}')
    #     di[:] = []
    #     print(di)
    #     print(fi)
    '''测试生成index_list'''
    # session_list = dict({
    #     1 : ['BENIGN','DDoS'],
    #     2 : ['DoS GoldenEye']
    # })
    # create_index_list('../data/index_list/cicids2017',session_list)
    '''调试CEC_read'''
    # imgs,labels = CEC_read_img('../data/index_list/cicids2017/session_1.txt')
    # print("imgs :\ntype:{}\ndata:{}\n".format(type(imgs[0]), imgs[:3]))
    # print("labels :\ntype:{}\ndata:{}\n".format(type(labels[0]), labels[:3]))
    '''test labels'''
    # label = ['cat', 'dog', 'bird', 'cat', 'dog', 'fish']
    # label2 = ['cat','hahahah']
    # le = LabelEncoder()
    # le.fit(label)
    # print('le 1 : \t{}'.format(le.classes_))
    # all_label = le.classes_.tolist() + label2
    # print(all_label)