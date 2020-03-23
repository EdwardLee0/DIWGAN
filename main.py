import  argparse 
import os
import sys
import tensorflow as tf
sys.path.extend([os.path.abspath("."), os.path.abspath("./../..")])
import inout_util as ut
from diwgan_model import  diwgan
os.chdir(os.getcwd() )
print('pwd : {}'.format(os.getcwd()))

parser = argparse.ArgumentParser(description='')
#set load directory
parser.add_argument('--image_path', dest='image_path', default=os.getcwd()+'/data', help='dicom file directory')
parser.add_argument('--input_path1', dest='input_path1', default='patchinput1', help='input image1 folder name')
parser.add_argument('--label_path1', dest='label_path1', default='patchlabel1', help='label image1 folder name')
parser.add_argument('--input_path2', dest='input_path2', default='patchinput2', help='input image2 folder name')
parser.add_argument('--label_path2', dest='label_path2', default='patchlabel2', help='label image2 folder name')
parser.add_argument('--test_image', dest='test_image', default= 'testinput')
parser.add_argument('--train_image', dest='train_image', default= 'traininput')

#set save directory
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',  default=os.getcwd()+'/checkpoint', help='check point dir')
parser.add_argument('--test_npy_save_dir', dest='test_npy_save_dir',  default=os.getcwd()+'/test', help='test numpy file save dir')

#image info
parser.add_argument('--patch_size', dest='patch_size', type=int,  default=128, help='image patch size, h=w')
parser.add_argument('--whole_size', dest='whole_size', type=int,  default=256, help='image whole size, h=w')
parser.add_argument('--img_channel', dest='img_channel', type=int,  default=1, help='image channel, 1')

#train, test
parser.add_argument('--model', dest='model', default='diwgan', help='red_cnn, cyclegan')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')

#train detail
parser.add_argument('--num_epoch', dest = 'num_epoch', type = int, default = 800, help = 'epoch')
parser.add_argument('--num_iter', dest = 'num_iter', type = int, default =4050, help = 'iterations')
parser.add_argument('--alpha', dest='alpha', type=float,  default=1e-4, help='learning rate')
parser.add_argument('--batch_size', dest='batch_size', type=int,  default=32, help='batch size')
parser.add_argument('--d_iters', dest='d_iters', type=int,  default=4, help='discriminator iteration') 
parser.add_argument('--lambda_1', dest='lambda_g1', type=int,  default=10, help='Gradient penalty term weight1')
parser.add_argument('--lambda_2', dest='lambda_g2', type=int,  default=10, help='Gradient penalty term weight2')
parser.add_argument('--lambda_w1', dest='lambda_w1', type=float,  default=0.01, help='D loss weight(in diwgan network)')
parser.add_argument('--lambda_w2', dest='lambda_w2', type=float,  default=0.01, help='D loss weight(in diwgan network)')
parser.add_argument('--lambda_3', dest='lambda_1', type=float,  default=1, help='l1 loss weight(in diwgan network)')
parser.add_argument('--lambda_4', dest='lambda_2', type=float,  default=1, help='l1 loss weight(in diwgan network)')
parser.add_argument('--lambda_gd1', dest='lambda_gd1', type=float,  default=0.5, help='gdl loss weight')
parser.add_argument('--lambda_gd2', dest='lambda_gd2', type=float,  default=0.5, help='gdl loss weight')
parser.add_argument('--lc1', dest='lc1', type=float,  default=10, help='g1_loss_final')
parser.add_argument('--lc2', dest='lc2', type=float,  default=8 , help='g2 loss_final')
parser.add_argument('--beta1', dest='beta1', type=float,  default=0.5, help='Adam optimizer parameter')
parser.add_argument('--beta2', dest='beta2', type=float,  default=0.9, help='Adam optimizer parameter')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=200, help='save a model every save_freq (iteration)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=1, help='print_freq (iterations)')
parser.add_argument('--continue_train', dest='continue_train', type=ut.ParseBoolean, default=True, help='load the latest model: true, false')
parser.add_argument('--gpu_no', dest='gpu_no', type=int,  default=0, help='gpu no')

# -------------------------------------
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
model = diwgan(sess, args)
model.train(args) if args.phase == 'train' else model.test(args)