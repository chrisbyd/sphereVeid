import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description="your script description")
    # the mode ['train','test']
    parser.add_argument('--mode', type=str, default='train')
    #train parameters
    parser.add_argument('--batch_size',type= int , default= 36)
    parser.add_argument('--batch_size_gen', type= int ,default= 16)
    parser.add_argument('--num_workers', type= int, default= 8)
    parser.add_argument('--num_instances', type= int, default= 6)
    parser.add_argument('--num_instances_gen', type=int, default=4)
    parser.add_argument('--encoder_middle_dim', type= int, default= 32)
    parser.add_argument('--optimizer_name',type=str, default='sgd')
    parser.add_argument('--dataset_name', type = str, default='veri')
    parser.add_argument('--dataset_root_dir', type=str, default='./datasets')
    parser.add_argument('--log_interval',type= int,default= 10)
    parser.add_argument('--train_epoches', type= int, default=100)

    #baseline model config
    parser.add_argument('--model_name',type= str, default= 'resnet50')
    parser.add_argument('--m_pretrain_path', type= str, default= './out/pretrained')
    parser.add_argument('--pretrain', type=bool, default= False)

    parser.add_argument('--train_generator_epoch', type=int, default= 30)

    #test parameters
    parser.add_argument('--test_batch_size', type=int, default=64)
    # whether to resume from stored model
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--resume_model_path', type=str, default='out/base/models/')
    parser.add_argument('--resume_epoch_number', type=str, default='0')

    # logger configuration
    parser.add_argument('--output_path', type=str, default='out/base/')
    parser.add_argument('--max_save_model_num', type=int, default=10, help='0 for max num is infinit')
    parser.add_argument('--save_images_path', type=str, default='out/base/' + 'images/')
    parser.add_argument('--save_models_path', type=str, default='out/base/' + 'models/')
    parser.add_argument('--save_features_path', type=str, default='out/base/' + 'features/')
    parser.add_argument('--save_model_interval', type=int, default=5)

    # test configuration
    parser.add_argument('--pretrained_model_path', type=str, default='out/base/' + 'models/')
    parser.add_argument('--pretrained_model_epoch', type=str, default='649')
    parser.add_argument('--test_interval', type=int, default=2)
    parser.add_argument('--view_gan_images_interval', type=int, default=20)

    args = parser.parse_args()
    return args
