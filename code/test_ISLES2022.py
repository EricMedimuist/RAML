import os
import argparse
import torch
import shutil
from networks.net_factory_3d import create_model
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from test_util import test_all_case,test_all_case_ACMT
from dataloaders.isles2022_process import *
import nibabel as nib
import SimpleITK as sitk
from medpy import metric
from scipy.stats import wilcoxon
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='ISLES2022', help='dataset_name')
parser.add_argument('--root_path', type=str, default='dataset', help='Name of Experiment')
parser.add_argument('--weight_path', type=str,
                    default='model/ISLES2022/vnet_200_labeled/', help='weight path')
parser.add_argument('--model', type=str,  default="vnet", help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
#模型权重根路径
snapshot_path = "./"
#模型权重路径
save_mode_path = os.path.join(snapshot_path,FLAGS.weight_path)

#预测结果保存根路径
test_root_path = "./prediction/"
#实验名称
test_exp_name = FLAGS.weight_path.split("/")[-2]
#测试结果保存路径
test_save_path = os.path.join(test_root_path, FLAGS.dataset_name, test_exp_name)+"/"

if os.path.exists(test_save_path):
    shutil.rmtree(test_save_path)
os.makedirs(test_save_path)

num_classes = 2

RAML_dice_40_labeled_percent=[0.8208008380863695, 0.8172661870503597, 0.7849462365591398, 0.8525021949078139, 0.764522417153996, 0.7294117647058823, 0.884429580081754, 0.8353626257278983, 0.8013468013468014, 0.8183141886426145, 0.8362492628268134, 0.07692307692307693, 0.7605956471935853, 0.658008658008658, 0.8729641693811075, 0.4049079754601227, 0.8413015388229766, 0.6229508196721312, 0.764070932922128, 0.6890756302521008, 0.91146408839779, 0.86980687374679, 0.8818225668806352, 0.8051514428809922, 0.74375, 0.6493055555555556, 0.7283950617283951, 0.6567349472377405, 0.7061224489795919, 0.8510366146138677, 0.883589329021827, 0.814663951120163, 0.0, 0.0, 0.6916382898221718, 0.8107656999791363, 0.9096832328273377, 0.7617421007685738, 0.8325320512820513, 0.9307944005707678, 0.7644305772230889, 0.31420765027322406, 0.818639798488665, 0.7846972529843234, 0.7217731421121252, 0.8544385718908424, 0.8426294820717132, 0.3348467650397276, 0.7691778465772274, 0.6789895255699322]

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)
    # assd=metric.binary.assd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)

    return dice, jc, precision, recall

def  Inference_my():

    net = create_model(name=FLAGS.model,in_chns=1, class_num=2,has_dropout=False)

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    "用于保存未归一化的原始图片"
    db_test_tmp = ISLES2022(base_dir='./dataset',
                            split='test',
                            transform_1=get_test_transform(),
                            )

    db_test = ISLES2022(base_dir='./dataset',
                        split='test',
                        transform_1=get_val_transform(),
                        )
    # print("len of  db_test:",len(db_test) )

    total_metric = np.zeros(4)
    metric_dice = []
    metric_jac = []
    metric_precision = []
    metric_recall = []

    metric_dice_std = []
    metric_jac_std  = []
    metric_precision_std  = []
    metric_recall_std  = []

    print("Testing begin")

    with open(test_save_path + "metric.txt", "a") as f:
        f.writelines("--------------test metric for {}-----------------\n".format(test_exp_name))

        "记录测试的样本个数"
        k=len(db_test)
        for i in range(len(db_test)):
            ids = i + 201
            print("test metric for sample_{} ".format(ids))

            "未做归一化的图片，[1, 1, 64, 64, 64]"
            image_tmp = db_test_tmp[i]['image'].as_tensor().unsqueeze(dim=0)

            "[1, 1, 64, 64, 64]"
            image = db_test[i]['image'].as_tensor().unsqueeze(dim=0)
            label = db_test[i]['label'].as_tensor().unsqueeze(dim=0)
            image, label =  image.cuda(), label.cuda()
            # print("image.shape:",image.shape)
            # print("label.shape:",label.shape)

            with torch.no_grad():
                output = net(image)
                if len(output)>1:
                    output=output[0]
                # ensemble
                output_soft = torch.softmax(output, dim=1)
                print("output_soft:",output_soft.shape)
                print("output_soft[0,0,0,0,0]:", output_soft[0,0,0,0,0])
                print("output_soft[0,1,0,0,0]:", output_soft[0, 1, 0, 0, 0])
                # print("output_soft[:, 0, ...].min():",output_soft[:, 0, ...].min())
                # print("output_soft[:, 0, ...].max():",output_soft[:, 0, ...].max())
                prediction = output_soft[:,1,...].unsqueeze(dim=1)
                print("prediction:",prediction.shape)
                # prediction = torch.argmax(output_soft, dim=1,keepdim=True)


            "[64, 64, 64]"
            prediction = prediction[0,0,...].cpu().data.numpy()
            # print(" prediction.shape:", prediction.shape)


            "[64, 64, 64]"
            image = image_tmp[0,0,...].cpu().data.numpy()
            # print(" image.shape:", image.shape)

            "[64, 64, 64]"
            label = label[0,0,...].cpu().data.numpy()
            # print(" label.shape:", label.shape)

            metric = calculate_metric_percase(prediction, label)
            print("metric:", metric)

            if metric[0]>0.1:
               metric_dice_std.append(metric[0])
               metric_jac_std.append(metric[1])
               metric_precision_std.append(metric[2])
               metric_recall_std.append(metric[3])

            total_metric[:] += metric
            metric_dice.append(metric[0])
            metric_jac.append(metric[1])
            metric_precision.append(metric[2])
            metric_recall.append(metric[3])



            f.writelines("Sample_{}:{},{},{},{}\n".format(
                ids, metric[0], metric[1], metric[2], metric[3]))

            pred_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, test_save_path +
                            "Sample_{}_pred.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, test_save_path +
                            "Sample_{}_img.nii.gz".format(ids))

            lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, test_save_path +
                            "Sample_{}_lab.nii.gz".format(ids))


        # average = total_metric / k
        # average = 100 * average
        dic_avg = np.average(metric_dice)
        jac_avg = np.average(metric_jac)
        pre_avg = np.average(metric_precision)
        rec_avg = np.average(metric_recall)
        avg = [dic_avg , jac_avg , pre_avg , rec_avg]

        avg1 = 100 * avg

        dic_std = np.std(metric_dice_std)
        jac_std = np.std(metric_jac_std)
        pre_std = np.std(metric_precision_std)
        rec_std = np.std(metric_recall_std)
        std=[dic_std,jac_std,pre_std,rec_std]
        std1 = 100 * std

        print("Mean metrics:{:.4f},{:.4f},{:.4f},{:.4f}".format(avg[0], avg[1], avg[2], avg[3]))
        print("std metrics:{:.4f},{:.4f},{:.4f},{:.4f}".format(std[0], std[1], std[2], std[3]))
        # print("Mean metrics:{:2.2f},{:2.2f},{:2.2f},{:2.2f}".format(100 *avg[0], 100 *avg[1], 100 *avg[2], 100 *avg[3]))
        # print("std metrics:{:2.2f},{:2.2f},{:2.2f},{:2.2f}".format(100 *std[0], 100 *std[1], 100 *std[2], 100 *std[3]))

        statistic, p_value = wilcoxon(RAML_dice_40_labeled_percent, metric_dice, zero_method='wilcox')
        print(f"Wilcoxon统计量: {statistic}")
        print(f"P值: {p_value}")

        f.writelines("Mean metrics:{:.4f},{:.4f},{:.4f},{:.4f}".format( avg[0] , avg[1] , avg[2] , avg[3] ))
        f.writelines("std metrics:{:.4f},{:.4f},{:.4f},{:.4f}".format(std[0], std[1], std[2], std[3]))
        f.writelines(f"Wilcoxon统计量: {statistic}")
        f.writelines(f"P值: {p_value}")
        # f.writelines("Mean metrics:{:2.2f},{:2.2f},{:2.2f},{:2.2f}".format(100 * avg[0], 100 * avg[1], 100 * avg[2],
        #                                                             100 * avg[3]))
        # f.writelines("std metrics:{:2.2f},{:2.2f},{:2.2f},{:2.2f}".format(100 * std[0], 100 * std[1], 100 * std[2],
        #                                                            100 * std[3]))

    f.close()
    print("Testing end")

    # std = [np.std(metric_dice), np.std(metric_jac), np.std(metric_hd), np.std(metric_asd)]
    return 0








if __name__ == '__main__':
    metric = Inference_my()

