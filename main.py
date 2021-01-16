import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import alexnet, fc
#from utils import get_id, get_data, accuracy
from utils import get_data, accuracy
from utils import get_grads, alpha_estimator, alpha_estimator2
from utils import linear_hinge_loss, get_layerWise_norms
import pdb
import numpy as np

def eval(eval_loader, net, crit, opt, layer_para_num, number, args, test=True):

    net.eval()

    # run over both test and train set    
    total_size = 0
    total_loss = 0
    total_acc = 0
    grads = []
    # outputs = []
    outputs = [0]

    P = 0 # num samples / batch size
    for x, y in eval_loader:
        P += 1
        # loop over dataset
        x, y = x.to(args.device), y.to(args.device)
        opt.zero_grad()
        # pdb.set_trace()
        out = net(x)
        
        # outputs.append(out)

        loss = crit(out, y)
        prec = accuracy(out, y)
        bs = x.size(0)

        loss.backward()
        grad = get_grads(net).cpu()
        grads.append(grad)

        total_size += int(bs)
        total_loss += float(loss) * bs
        total_acc += float(prec) * bs

    M = len(grads[0]) # total number of parameters
    grads = torch.cat(grads).view(-1, M)
    mean_grad = grads.sum(0) / P

    # pdb.set_trace()
    norm_full_grad = mean_grad.norm()
    norm_minibatch_grad = grads.norm(dim=1).mean()

    grads = grads - mean_grad

    norm_minibatch_grad_noise = grads.norm(dim=1).mean()
    # pdb.set_trace()
    noise_norm = torch.cat((norm_minibatch_grad_noise.unsqueeze(0), norm_minibatch_grad.unsqueeze(0), norm_full_grad.unsqueeze(0)), 0)

    N = M * P
    for i in range(1, 1 + int(math.sqrt(N))):
        if N % i == 0:
            m = i
    # pdb.set_trace()
    alpha = alpha_estimator(m, grads.view(-1, 1))
    # if number == 22:
    #     kkk = 0
    #     pdb.set_trace()
    # pdb.set_trace()
    del grads
    del mean_grad

    hist = [ total_loss / total_size, total_acc / total_size, alpha.item()] #+ noise_alphas3

    # pdb.set_trace()
    print(np.round(hist,2))
    
    return hist, outputs, noise_norm


#CUDA_VISIBLE_DEVICES=0 python mainadam4.py --dataset cifar10 --model fc --depth 7  --width 512 --optimizer sgd --lr 0.05

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_eval', default=100, type=int,
                        help='must be equal to training batch size')
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--print_freq', default=200, type=int)
    parser.add_argument('--eval_freq', default=200000000, type=int)
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='mnist | cifar10 | cifar100')
    parser.add_argument('--path', default='./data', type=str)
    parser.add_argument('--seed', default=0, type=int)
    #parser.add_argument('--model', default='fc', type=str)alexnet
    parser.add_argument('--model', default='fc', type=str)
    parser.add_argument('--criterion', default='NLL', type=str,
                        help='NLL | linear_hinge')
    parser.add_argument('--scale', default=64, type=int,
                        help='scale of the number of convolutional filters')
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--width', default=100, type=int,
                        help='width of fully connected layers')
    parser.add_argument('--save_dir', default='model/', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--lr_schedule', action='store_true', default=False)
    args = parser.parse_args()

    # initial setup
    if args.double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    
    print(args)

    # training setup
    train_loader, test_loader_eval, train_loader_eval, num_classes = get_data(args)

    if args.model == 'fc':
        if args.dataset == 'mnist':
            net = fc(width=args.width, depth=args.depth, num_classes=num_classes).to(args.device)
        elif args.dataset == 'cifar10':
            net = fc(width=args.width, depth=args.depth, num_classes=num_classes, input_dim=3*32*32).to(args.device)
        elif args.dataset == 'cifar100':
            net = fc(width=args.width, depth=args.depth, num_classes=num_classes, input_dim=3*32*32).to(args.device)
    elif args.model == 'alexnet':
        net = alexnet(ch=args.scale, num_classes=num_classes).to(args.device)

    print(net)

    if args.optimizer == 'sgd' or args.optimizer == 'sgdm':
        opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        opt = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)

    if args.lr_schedule:
        milestone = int(args.iterations / 4)
        scheduler = optim.lr_scheduler.MultiStepLR(opt, 
            milestones=[milestone, 2*milestone],
            gamma=0.5)
    
    if args.criterion == 'NLL':
        crit = nn.CrossEntropyLoss().to(args.device)
    elif args.criterion == 'linear_hinge':
        crit = linear_hinge_loss
    
    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)
    
    # training logs per iteration
    training_history = []
    weight_grad_history = []

    layer_para_num = [param.numel() for param in net.parameters()]

    # eval logs less frequently
    evaluation_history_TEST = []
    evaluation_history_TRAIN = []
    noise_norm_history_TEST = []
    noise_norm_history_TRAIN = []

    STOP = False
    number = 0
    for i, (x, y) in enumerate(circ_train_loader):

        if i % args.eval_freq == 0:
            # first record is at the initial point

            te_hist, te_outputs, te_noise_norm = eval(test_loader_eval, net, crit, opt, layer_para_num, number, args)
            tr_hist, tr_outputs, tr_noise_norm = eval(train_loader_eval, net, crit, opt, layer_para_num, number, args,
                                                      test=False)
            evaluation_history_TEST.append([i, *te_hist])
            evaluation_history_TRAIN.append([i, *tr_hist])
            noise_norm_history_TEST.append(te_noise_norm)
            noise_norm_history_TRAIN.append(tr_noise_norm)

            number += 1

        net.train()
        
        x, y = x.to(args.device), y.to(args.device)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        # calculate the gradients
        loss.backward()

        # record training history (starts at initial point)
        training_history.append([i, loss.item(), accuracy(out, y).item()])
        # weight_grad_history.append([i, *get_layerWise_norms(net)])

        # take the step
        opt.step()

        if i % args.print_freq == 0:
            print(training_history[-1])

        if args.lr_schedule:
            scheduler.step(i)



        if (i<=3000 and i % 50 == 0) or (i>3000 and i % 500 == 0):
            # final evaluation and saving results
            print('eval time {}'.format(i))
            te_hist, te_outputs, te_noise_norm = eval(test_loader_eval, net, crit, opt, layer_para_num,  number, args)
            tr_hist, tr_outputs, tr_noise_norm = eval(train_loader_eval, net, crit, opt, layer_para_num,  number, args, test=False)
            evaluation_history_TEST.append([i + 1, *te_hist])
            evaluation_history_TRAIN.append([i + 1, *tr_hist])
            noise_norm_history_TEST.append(te_noise_norm)
            noise_norm_history_TRAIN.append(tr_noise_norm)

        if (i > 30000 and i % 5000 == 0) or i <=1:
            # pdb.set_trace()
            path_save = ('%s/new_%s_%s_%s_%d_%d_%.5f_%.2f_%d_%d_%d/'%(args.save_dir,args.dataset,args.optimizer,args.model,args.depth,args.width,args.lr,args.mom,args.batch_size_train,args.iterations,args.lr_schedule))

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            else:
                print('Folder already exists, beware of overriding old data!')

            # save the setup
            torch.save(args, path_save + '/args.info')
            # save the outputs
            torch.save(te_outputs, path_save + '/te_outputs.pyT')
            torch.save(tr_outputs, path_save+ '/tr_outputs.pyT')
            # save the model
            state = {'net':net.state_dict(), 'optimizer':opt.state_dict(),'epoch':i}
            torch.save(state, path_save + '/state_%d.pyT'%i)
            # save the logs
            torch.save(training_history, path_save + '/training_history.hist')
            torch.save(evaluation_history_TEST, path_save + '/evaluation_history_TEST.hist')
            torch.save(evaluation_history_TRAIN, path_save + '/evaluation_history_TRAIN.hist')
            torch.save(noise_norm_history_TEST, path_save + '/noise_norm_history_TEST.hist')
            torch.save(noise_norm_history_TRAIN, path_save + '/noise_norm_history_TRAIN.hist')

        if i > args.iterations:
            break