```相对的改进
#---------------------------(predict.ipynb)------------------------#
    if mode == "predict":
        image = Image.open('4.jpg')#先自动对目录下的4.jpg文件实施基线预测
        #此外还提供的基准图片有：45是基线目标，67是大目标，89是小目标，cd是难目标
        r_image = yolo.detect_image(image, crop = crop, count=count)
        r_image.show()
        while True:
            img = input('Input image filename:')
            try:#这样自动叠加后缀就只需要输入文件名
                image = Image.open(img+'.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
#---------------------------(yolo.py)------------------------#
        "model_path"        : 'model_data/b基础633.pth',#原yolov8_s换为自训的基线权
        "classes_path"      : 'model_data/voc_classes.txt',#只含0到6七类，分别分行
        "phi"               : 'n',#版本从s换为更易训、内存更小的n 
        "cuda"              : False,#cuda换为否方便推理时切无卡模式用cpu更省钱
#---------------------------(utils_fit.py)------------------------#
    if local_rank == 0:#去掉开训和完训，以及验证全程的显示
        # print('Start Train')
    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):#关掉最优权的保存提示，将定期权重名改为p030三个数的形式，忽略具体损失，最后精简best_epoch_weights为b，last_epoch_weights为l
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
        #     torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
#---------------------------(callbacks.py)------------------------#
            # print("Calculate Map.")
            # print("Get map done.") #关掉算map始末的提示
#---------------------------(train.ipynb)------------------------#
if __name__ == "__main__": #精简参数行，去除多余注释
    Cuda            = True #服务器训练只能用gpu，无卡模式cpu训不了
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True #设true更快些
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'b基础633.pth' #原为'model_data/yolov8_s.pth'改成咱们自训的
    input_shape     = [640, 640]
    phi             = 'n' # 原's'改更小更高效
    pretrained      = False #有权重就不用预训练
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2 #原32改小
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4 #原16改小
    Freeze_Train        = False #预冻结前50的骨网权重，在前置网需要同时训练故设False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 30 #每隔30轮保存下权重，整个只需10个文件，减少原10的冗余
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 4
#---------------------------(voc_annotation.py)------------------------#
annotation_mode     = 2 #基本的集合已被划分于ImageSeg文件夹，现只需生成2007_train.txt、2007_val.txt的目标信息即可（原为0）
#-----------------------------------(utils_map.py)------------------------------#
# 第241和第609行，均加入".manager"变为fig.canvas.manager.set_window_title
    fig.canvas.manager.set_window_title(window_title)#第241行
                fig.canvas.manager.set_window_title('AP ' + class_name)#609行
```
损失部分：由于只改动位损，在推理时可以直接载入权值文件，因此不附加权值，loss辅和loss[3]均已初化为零，两次训练的map值（终值算一，峰值算半）：
原：678/643
后：654(-2+3=1)/715(7+1=8)
```
#---------------------------(yolo_training.py，类损)----------------------------#
class Loss:
    def __call__(): #若不想加此损，此处loss辅直改0
        # loss[1]=self.bce(pred_scores,target_scores.to(dtype)).sum()/target_scores_sum #BCE  #先在末维生成等差矩阵(b,8400,7) ↓
        emb=torch.Tensor(np.arange(self.nc)).float().to(device).view(1,1,-1)
        loss辅=(torch.sum(torch.abs(torch.argmax(target_scores,-1,True).repeat(1,1,7)*torch.ones_like(target_scores)-emb)*F.softmax(pred_scores,-1),-1)*torch.sum(target_scores,-1)).sum()/target_scores_sum #具体见如下公式 ↓
        loss[1]=loss[1]+loss辅 #Lshape=Σ{k=0→nc}||k-c_right||·softmax(P_0:nc)
#------------------------------(utils_bbox.py,后处理)----------------------------#
def non_max_suppression():# 51.37→51.5，效果虽微弱但仍有提升
            # class_conf,class_pred=torch.max()
            x=image_pred[:,4:4+num_classes]
            class_pred=torch.round(torch.sum((x/torch.sum(x,1,True))*torch.arange(0,7,device=x.device).unsqueeze(0).expand(8400,7),dim=1)).unsqueeze(1)
            class_conf=torch.gather(x, 1, class_pred.long())

#---------------------------(yolo_training.py，位损)----------------------------#
import os,torchvision,random##最初506
from torchvision.utils import save_image
class Loss: # 第368行
    def __call__():
        # device  = preds[1].device
        loss    = torch.zeros(4, device=device) #后面到loss[3]一次性添入
        # pred_bboxes = self.bbox_decode(anchor_points,pred_distri)#xyxy,(b,h*w,4)
        低框=pred_bboxes[:,:6400,:].view(-1,4,80,80)#拆为(bs,6400,4)转(bs,4,80,80)
        框分=torch.max(pred_scores,-1,keepdim=False)[0]#(b,6k,nc)转(b,6k)
        信权=(框分[:,:6400].sigmoid()).view(-1,80,80).unsqueeze(1); 权重=信权.sum()
        # 拟框=F.avg_pool2d(低框*信权,3,1,1)/F.avg_pool2d(信权,3,1,1)
        拟框=F.avg_pool2d(低框,3,1,1) #原计划是用上面一行，但本行已证亦可
        if random.random()<0.1:  #取低框激活转(b,80,80)↑
            扩信 = F.interpolate(信权,size=(640,640)).squeeze(1)
            image=torchvision.transforms.ToPILImage()(((扩信[0]>0.01)*255).byte())
            image.save('logs/ctt.png')#保存首图的信区得到一个类掩码图于日志目录下
        loss[3] = F.smooth_l1_loss(低框*信权, 拟框*信权)/权重#不想加此损，此处直改0
        return loss.sum(),loss[3] #最后输出补上该邻损显示于实时的进度条
#---------------------------(utils_fit.py)----------------------------#
def fit_one_epoch():
    # val_loss    = 0
    anl =0
        # if not fp16:
        # else:
        #         outputs         = model_train(images)
                loss_value,loss_angle = yolo_loss(outputs, bboxes)
        loss += loss_value.item()
        # anl  += loss_angle.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss':loss/(iteration+1),"结":anl/(iteration+1),'lr':get_lr(optimizer)})
        #    验证时的前向传播
        val_loss += loss_value[0].item()
        
#------------------------(yolo_training.py，早期位损)----------------------------#
import os,torchvision,random##最初506
from torchvision.utils import save_image
class Loss: # 第368行
    def __call__():
        # device  = preds[1].device
        loss    = torch.zeros(4, device=device)
        # pred_bboxes = self.bbox_decode(anchor_points,pred_distri)#xyxy,(b,h*w,4)
        低框=pred_bboxes[:,:6400,:].view(-1,80,80,4)#拆为(bs,6400,4)转(bs,80,80,4)
        框分=torch.max(pred_scores,-1,keepdim=False)[0]#(b,6k,nc)转(b,6k)
        信权=(框分[:,:6400].sigmoid()).view(-1,80,80).unsqueeze(1); 权重=信权.sum()
        if random.random()<0.1:  #取低框激活转(b,80,80)↑
            扩信 = F.interpolate(信权,size=(640,640)).squeeze(1)
            image=torchvision.transforms.ToPILImage()(((扩信[0]>0.01)*255).byte())
            image.save('logs/ctt.png')#保存首图的信区得到一个类掩码图于日志目录下
        差损 = F.smooth_l1_loss(邻损(低框)*信权, torch.zeros_like(邻损(低框)*信权))
        loss[3] = 差损*5000000/权重 #如果不想添加这项损失，此处直接改作0
        return loss.sum(),loss[3] #最后输出补上该邻损显示于实时的进度条
def 交损(box1, box2, threhold=0.5):#两(b,w,h,4)→(b,w,h)
    x1=torch.max(box1[...,0],box2[...,0]); y1=torch.max(box1[...,1],box2[...,1])
    x2=torch.min(box1[...,2],box2[...,2]); y2=torch.min(box1[...,3],box2[...,3])
    intersection=torch.clamp(x2-x1,min=0)*torch.clamp(y2-y1,min=0)
    area1=(box1[...,2]-box1[...,0])*(box1[...,3]-box1[...,1])
    area2=(box2[...,2]-box2[...,0])*(box2[...,3]-box2[...,1])
    return 1-intersection / (area1 + area2 - intersection)#改0.001
def 邻损(当):#(b,w,h,4)→(b,w,h)
    右,下 = 当*0,当*0; 右[:,:-1,:,:]=当[:,1:,:,:]; 下[:,:,:-1,:]=当[:,:,1:,:]
    return 交损(当,右)+交损(当,下)
```