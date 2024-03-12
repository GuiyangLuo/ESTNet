import torch
import numpy as np
import argparse
import time
import utils.util  as util
import matplotlib.pyplot as plt
from utils.engine import trainer
import os
import torch.nn as nn
from datetime import datetime
from tensorboardX  import SummaryWriter
import re

def find_nearest_in_dir(model_dir = './garage'):
    import os
    files = os.listdir(model_dir)
    min = -1
    min_file = ''
    for file in files:
        cur = file.split('_')[1]
        if not cur.isdigit():
            continue
        cur = int(cur)
        if cur > min:
            min = cur
            min_file = file
    if min <  0:
        return None,None
    return  os.path.join(model_dir, min_file), min


def find_bestmodel_in_dir(model_dir = './garage'):
    import os
    files = os.listdir(model_dir)
    min = 1000
    min_file = ''
    for file in files:
        cur = file.split('_')[2]
        cur = re.findall(r"\d+\.?\d*", cur)[0]
        if not cur.replace(".", "").isdigit():
            continue
        cur = float(cur)
        if cur < min:
            min = cur
            min_file = file
    return  os.path.join(model_dir, min_file)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(" haha --------------------------------------")
        # print(" haha ---", classname, m, m.weight, m.bias, )
        # print(" haha ---", classname, m, m.weight.data.shape, m.bias.data.shape, )
        nn.init.xavier_uniform_(m.weight.data,  gain=nn.init.calculate_gain('relu'))

def train(args):

    device = torch.device(args.device)
    if args.use_tensorboard:
        tensorboard_dir_final = os.path.join(args.tensorboard_dir,
                                             'runs_{}_{}'.format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')[:-3],
                                                                 args.exp_id))
        if not os.path.exists(tensorboard_dir_final):
            os.makedirs(tensorboard_dir_final)
        writer = SummaryWriter(tensorboard_dir_final)


    dataloader = util.load_dataset(args.data_output_path, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    engine = trainer(scaler, args, writer = writer)
    engine.model.apply(weights_init)
    # for p in engine.model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)
    #     else:
    #         nn.init.uniform_(p)

    trainable_parameters = []
    for name, param in engine.model.named_parameters():
        if param.requires_grad:
            trainable_parameters.append(name)
    model_save_dir = args.model_save_dir + str(args.exp_id)
    epoch_existing = None
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    else:
        model_path , epoch_existing = find_nearest_in_dir(model_save_dir)
        if model_path is not None:
            print("Load the existing model ", model_path)
            engine.model.load_state_dict(torch.load(model_path))





    print("Start training...",flush=True)
    # print("-------------------All the trainable parameters are: ---------------------")
    # print(trainable_parameters)
    print("---------------------The total number of parameters is ---------------------")
    print("Total parameters: ", sum(p.numel() for p in engine.model.parameters() if p.requires_grad))
    # for name, p in engine.model.named_parameters():
    #     print(name, p.shape,p.numel())

    file_name = os.path.join(args.model_save_dir, 'parameters.txt')
    if not os.path.exists(file_name):
        os.mknod(file_name)
    with open(file_name,'w') as f:
        for name, p in engine.model.named_parameters():
            string_data = str(name)+ "  "+ str( p.shape) +"  "+ '\n'
            f.write(string_data)
        f.write(" + \n------------- + \n ")
        f.write(str(args))
    print( "  model  graph and parameters are saved into ",str(file_name))

    print("------------------------------------------------------------------------")


    his_loss =[]
    val_time = []
    train_time = []
    global_iter = 0
    begin_epoch = 1 if epoch_existing is None else epoch_existing

    for i in range(begin_epoch,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            # print('trainx',x.shape, y.shape)
            # trainx shape: (batch_size, seq_len, numberofnodes , in_dims)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            metrics = engine.train(trainx, trainy[:,0,:,:])

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)

        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])


        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        if args.use_tensorboard:
            if i  == 1:
                writer.add_graph(engine.model, trainx)
            writer.add_scalar('Loss/loss', mtrain_loss, i)
            writer.add_scalar('Loss/mape', mtrain_mape, i)
            writer.add_scalar('Loss/rmse', mtrain_rmse, i)
            writer.add_scalar('Val_loss/loss', mvalid_loss, i)
            writer.add_scalar('Val_loss/mape', mvalid_mape, i)
            writer.add_scalar('Val_loss/rmse', mvalid_rmse, i)


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)

        print('save model of ',  model_save_dir+"/epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

        torch.save(engine.model.state_dict(), model_save_dir+"/epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        if i == begin_epoch:
            for _ in range(begin_epoch):
                engine.StepLR()
        engine.StepLR()
        print("第%d个epoch的学习率：%f" % (i, engine.optimizer.param_groups[0]['lr']))


    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    if epoch_existing is None:
        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(torch.load(model_save_dir+"/epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
        print("Training finished")
        print("The valid loss on best model is", model_save_dir+"/epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth")
    else:
        file = find_bestmodel_in_dir(model_save_dir)

        engine.model.load_state_dict(torch.load(file))
        torch.save(engine.model.state_dict(),
                   model_save_dir + "_final_best_" + ".pth")
        print("Training finished")
        print("The valid loss on best model is", file)



    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            engine.model.eval()
            preds = engine.model(testx)[0].transpose(1,3)

        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    amae = []
    amape = []
    armse = []

    for i in range(args.seq_length_y):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]

        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:2d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 4 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))

    return amae, amape, armse
