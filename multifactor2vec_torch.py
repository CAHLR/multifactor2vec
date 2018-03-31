__author__ = 'jwj'
import torch
import numpy as np
import pickle
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import sys
import getopt



try:
    opts, args = getopt.getopt(sys.argv[1:],'hi:o:f:e:d:m:b:l:v:t:s:',)
except getopt.GetoptError:
    print('\nxxx.py -i <inputfile> -o <outputfile> -c <coursefile> -f <factorfile> -e <epoch num> -d <vector dimension> -m <whether use minibatch> -b <minibatch size> -l <learning rate> -v <whether need validation>  -t <validationfile> \nDefault parameter: epoch=10, vector dimension=300, learning_rate=1e-3, minibatch=4096 \n')
    sys.exit(2)
for opt, arg in opts:
    if opt == "-h":
        print('\nxxx.py -i <inputfile> -o <outputfile> -c <coursefile> -f <factorfile> -e <epoch num> -d <vector dimension> -m <whether use minibatch> -b <minibatch size> -l <learning rate> -v <whether need validation>  -t <validationfile> \nDefault parameter: epoch=10, vector dimension=300, learning_rate=1e-3, minibatch=4096 \n')
        sys.exit()
    elif opt in ("-i", "--infile"):
        sampled_data = arg
    elif opt in ("-o", "--outfile"):
        saved_model = arg
    elif opt in ("-c", '--coursefile'):
        course_file = arg
    elif opt in ("-f", '--factorfile'):
        factor_file = arg
    elif opt == "-e":  # epoch: number of epoch
        epoch = int(arg)
    elif opt == "-d":  # vector dimension
        vector_dim = int(arg)
    elif opt == "-m":  # whether train on batch or mini-batch
        mini_batch = int(arg)
    elif opt == "-b":
        batch_s = int(arg)
    elif opt == "-l":
        learning_rate = int(arg)
    elif opt == "-v":  # whether need validation set to calculate valdiation loss
        need_validation = int(arg)
    elif opt in ("-t", "--validationfile"):
        validation_loss_file = arg
    elif opt == "-s":  # pure course2vec or multifactor2vec
        multi = int(arg)





class Net(torch.nn.Module):
    def __init__(self, idim, vector_dim, odim):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(idim, vector_dim, bias=False).cuda()
        self.l2 = torch.nn.Linear(vector_dim, odim, bias=False).cuda()
    def forward(self, x):
        h = self.l1(x)
        y_pred = self.l2(h)
        return y_pred




def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    index = np.random.randint(ylen, size=batch_size)
    target_batch = x[index]
    context_batch = y[index]
    target = np.zeros((batch_size, indim))
    for i in range(batch_size):
        target[i, target_batch[i]] = 1
    context_course = zip(*context_batch)
    context = list(context_course)[0]
# remove letter & science major
    target = np.concatenate((target[:, :7487], target[:, 7488:]), axis=1)
    return target, context


def train_batch():
    for t in range(10):  # epoch
        #scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        #scheduler.step()
        optimizer.zero_grad()
        for i in range(iter):
            target, context = generate_batch_data_random(word_target, word_context, batch_s)
            target = torch.FloatTensor(target)
            context = torch.LongTensor(context)
            target = Variable(target, requires_grad=False)
            target = target.cuda()
            context = Variable(context, requires_grad=False)
            context = context.cuda()
            y_pred = model(target).cuda()
            loss = loss_fn(y_pred, context)
            loss_fn.cuda()
            loss.backward()
            print('epoch' + str(t+1)+':' + 'The'+str(i+1)+'-th interation: training loss'+str(loss.data[0])+'\n')
            #optimizer.zero_grad()
        model.l1.weight.grad = model.l1.weight.grad / float(iter)
        model.l2.weight.grad = model.l2.weight.grad / float(iter)
        optimizer.step()

        if need_validation == True:
            validation_set_loss(t)


def train_mini_batch():
    for t in range(epoch):  # epoch
        #scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        #scheduler.step()
        for i in range(iter):
            target, context = generate_batch_data_random(word_target, word_context, batch_s)
            target = torch.FloatTensor(target)
            context = torch.LongTensor(context)
            target = Variable(target, requires_grad=False)
            target = target.cuda()
            context = Variable(context, requires_grad=False)
            context = context.cuda()
            optimizer.zero_grad()
            y_pred = model(target).cuda()
            loss = loss_fn(y_pred, context)
            loss_fn.cuda()
            print('epoch' + str(t+1)+':' + 'The'+str(i+1)+'-th interation: training loss'+str(loss.data[0])+'\n')
            loss.backward()
            optimizer.step()

        if need_validation == True:
            validation_set_loss(t)


# batch_size>16384, out of memory
def train_mini_batch1():
    for t in range(epoch):  # epoch
        #scheduler = MultiStepLR(optimizer, milestones=[5,6,7,8,9], gamma=0.1)
        #scheduler.step()
        for i in range(iter):
            optimizer.zero_grad()
            for j in range(iter1):
                target, context = generate_batch_data_random(word_target, word_context, 1024)
                target = torch.FloatTensor(target)
                context = torch.LongTensor(context)
                target = Variable(target, requires_grad=False)
                target = target.cuda()
                context = Variable(context, requires_grad=False)
                context = context.cuda()
                y_pred = model(target).cuda()
                loss = loss_fn(y_pred, context)
                loss_fn.cuda()
                loss.backward()
            print('epoch' + str(t+1)+':' + 'The'+str(i+1)+'-th interation: training loss'+str(loss.data[0])+'\n')
            model.l1.weight.grad = model.l1.weight.grad / float(iter1)
            model.l2.weight.grad = model.l2.weight.grad / float(iter1)
            optimizer.step()
        if need_validation == True:
            validation_set_loss(t)


def validation_set_loss(t):
    print('validation_loss_calculating')
    iter = vali_length // 4096
    print(iter)
    vali_loss_epoch = 0
    for i in range(iter+1):
        print(i)
        if i == iter:
            target_vali = word_target_vali[iter*4096:]
            context_vali = word_context_vali[iter*4096:]
        else:
            target_vali = word_target_vali[i:i+4096]
            context_vali = word_context_vali[i:i+4096]
        vali_batch_length = len(target_vali)
        target_vali_np = np.zeros((vali_batch_length, indim))
        for j in range(vali_batch_length):
            target_vali_np[j, target_vali[j]] = 1
        context_vali_np = zip(*context_vali)
        context_vali_np = list(context_vali_np)[0]
        target_vali_np = torch.FloatTensor(target_vali_np)
        context_vali_np = torch.LongTensor(context_vali_np)
        target_vali = Variable(target_vali_np, volatile=True).cuda()
        context_vali = Variable(context_vali_np, volatile=True).cuda()

        context_vali_pred = model(target_vali).cuda()
        vali_loss_iter = loss_fn(context_vali_pred, context_vali).cuda()
        vali_loss_epoch += vali_loss_iter.data[0] * vali_batch_length
    vali_loss_epoch /= vali_length
    print('epoch' + str(t+1)+': validation loss'+str(vali_loss_epoch)+'\n')
    f = open(validation_loss_file, 'a')
    f.write(str(vali_loss_epoch)+',')
    f.close()




if __name__ == '__main__':
    mini_batch = 1  # 0: on batch, other: mini-batch
    if need_validation == 0:
        need_validation = False
    else:
        need_validation = True

    course_file = open(course_file, 'rb')
    course = pickle.load(course_file)
    course_id = course['course_id']
    vocab_size = len(course_id)
    id_course = course['id_course']
    factor_file = open(factor_file, 'rb')
    major = pickle.load(factor_file)
    major_id = major['major_id']
    id_major = major['id_major']
    course_file.close()

    f = open(sampled_data, 'rb')
    data = pickle.load(f)
    f.close()
    word_target = list(data['course_target'])
    word_context = list(data['course_context'])
    word_target = np.array(word_target)
    word_context = np.array(word_context)
    print("construct model")

    if multi == 0:
        indim = len(course_id)
        outdim = indim
        data_length = len(word_target)
    else:
        indim = len(course_id) + len(major_id)
        outdim = len(course_id)
        data_length = len(word_target)

    if need_validation == True:
        # separate validation set and training set
        vali_length = data_length // 10
        word_target_vali = word_target[:vali_length]
        word_context_vali = word_context[:vali_length]
        word_target = word_target[vali_length:]
        word_context = word_context[vali_length:]
        vali_loss1 = list()

    loss_fn = torch.nn.CrossEntropyLoss()
    model = Net(indim, vector_dim, outdim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter1 = batch_s//1024
    iter = len(word_target)//batch_s

    if mini_batch == 0:
        print('train on batch')
        train_batch()
    elif mini_batch != 0 and batch_s <= 16384:
        print('train on mini_batch')
        train_mini_batch()
    elif mini_batch != 0 and batch_s > 16384:
        print('train on mini_batch>16384')
        train_mini_batch1()
    torch.save(model, saved_model)
