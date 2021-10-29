for dimention in dimentions:
    acc=[]
    for subject in subjects:  
        import numpy as np
        import os
        import argparse
        parser = argparse.ArgumentParser(description="Capsule Network on " + dataset_name)
        parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
        parser.add_argument('--epochs', default=10, type=int)  
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument('--lam_regularize', default=0.0, type=float,
                            help="The coefficient for the regularizers")
        parser.add_argument('-r', '--routings', default=3, type=int,
                            help="Number of iterations used in routing algorithm. should > 0")
        parser.add_argument('--debug', default=0, type=int,
                            help="Save weights by TensorBoard")
        parser.add_argument('--save_dir', default='./result_lr0.00001_'+ dataset_name + '/label_'+dimention+'/sub_dependent_'+subject+'/') # other
        parser.add_argument('-t', '--testing', action='store_true',
                            help="Test the trained model on testing dataset")
        parser.add_argument('-w', '--weights', default=None,
                            help="The path of the saved weights. Should be specified when testing")
        parser.add_argument('--lr', default=0.000001, type=float,
                            help="Initial learning rate")  # v0:0.0001, v2:0.00001
        # parser.add_argument('--lam_regularize', default=0.0, type=float,
                            # help="The coefficient for the regularizers")
        # parser.add_argument('--gpus', default=2, type=int)
        parser.add_argument('--adj' , default='psd', help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--save_freq'   , default=10, type=int, help='Save frequency')
        parser.add_argument('--model'       , default='GCN',      help='model') 
        args = parser.parse_args()
        GPU_IDX=0            
        DEVICE = torch.device('cuda:{}'.format(GPU_IDX) if torch.cuda.is_available() else 'cpu')
        print(time.asctime(time.localtime(time.time())))
        print(args)

        checkpoint_dir = '%s%s/%s/checkpoints'%(args.save_dir,args.model,args.adj)
        result_dir = '%s%s/%s/acc'%(args.save_dir,args.model,args.adj)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)  
        if not os.path.exists(result_dir):
            os.makedirs(result_dir) 
        if dataset_name == 'dreamer':         
            # load dreamer data
            datasets,labels,psd_weight = deap_load(subject,dimention,debaseline)
        else:  # load deap data
            datasets,labels,psd_weight = deap_load(subject,dimention,debaseline)
        
       #进行图粗化
        if args.adj=='distance':
            adj=np.load("../deap_shuffled_data/yes_dominance/adj_distances.npy",allow_pickle=True).astype('float32')
            adj=sparse.csr_matrix(adj)
            graphs,perm,parents=coarsening.coarsen(adj,levels=5,self_connections=False)
            datasets=coarsening.perm_data(datasets,perm)
            datasets=datasets.reshape(-1,32,128)
            #caculate L
            L=[graph.laplacian(adj,normalized=True).todense() for adj in graphs]
        if args.adj=='psd':
            # np.set_printoptions(threshold=np.inf)
            adj=psd_weight.reshape(32,32)
            adj=sparse.csr_matrix(adj)
            graphs,perms,parents=coarsening.coarsen(adj,datasets,levels=5,self_connections=False)
            datasets=coarsening.perm_data(datasets,perms[0])
            data_perm=[list(parent) for parent in parents]
            L=[graph.laplacian(adj,normalized=True).todense() for adj in graphs[:-1]]
       
            
        fold = 10
        test_accuracy_allfold =[]
        for curr_fold in range(10): 
            fold_size = datasets.shape[0] // fold
            print("fold_size:",fold_size)
            indexes_list = [i for i in range(len(datasets))]
            split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
            test_split = np.array(split_list)
            train_split = np.array(list(set(indexes_list) ^ set(split_list)))
        
            Train=EEGGraphDataset(datasets,labels,indices=train_split,transform=Compose([ToTensor()]))
            Trainloader=DataLoader(Train,batch_size=args.batch_size,shuffle=False)
            Test=EEGGraphDataset(datasets,labels,indices=test_split,transform=Compose([ToTensor()]))
            Testloader=DataLoader(Test,batch_size=args.batch_size,shuffle=False)

            model=EEGGraphConvNet(L,data_perm,input_dim=128,
                                F=[256,512,128,32,16],
                                K=[2,2,2,2,2],
                                P=[2,2,2,2,2],
                                n_layers=5,
                                batch_size=args.batch_size).to(DEVICE)
            print("model:",model)
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Total:', total_num)
            print('Trainable:',trainable_num)
            criterion=nn.CrossEntropyLoss()
            # criterion=nn.BCELoss()
            # train
            train_start_time = time.time()
            best_acc=0
            for epoch in (range(1,args.epochs+1)):
                correct=[]
                num_of_true=0
                optimizer = optim.Adam( model.parameters(),lr=args.lr,weight_decay=1e-2)
                for i,data in enumerate(tqdm(Trainloader)):
                    # label_indeices=np.array(data.y)
                    model.train()
                    data=data.to(DEVICE,non_blocking=True)                        
                    label=torch.as_tensor(data.y,dtype=torch.long).to(DEVICE)
                    optimizer.zero_grad()
                    output=model(data.x)
                    loss=criterion(output,label)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(output.cpu().data, 1)
                    num_of_true+=(predicted==label.cpu()).sum().item()
                loss=loss.item()
                train_acc=num_of_true/len(Trainloader.dataset)
                print(' subject:{} | curr_fold:{} | Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} train_acc: {}'.format(subject,curr_fold,epoch,i * len(data), len(Trainloader.dataset),100. * i / len(Trainloader),loss, train_acc))
                #valid
                test_loss=0
                correct=[]
                num_of_true=0
                model.eval()
                for i,data in enumerate(Testloader,0):
                    data=data.to(DEVICE,non_blocking=True)
                    label=torch.as_tensor(data.y,dtype=torch.long).to(DEVICE)
                    output=model(data.x)
                    test_loss+=criterion(output,label).item()
                    _, predicted = torch.max(output.cpu().data, 1)
                    num_of_true+= (predicted == label.cpu()).sum().item()
                test_loss /= (i+1)
                test_acc_fold=num_of_true/len(Testloader.dataset)
                print('(' + time.asctime(time.localtime(time.time())) + ') subject:',subject,'Test loss:',test_loss,'Test acc:', test_acc_fold )
                print('-' * 30 +str(subject) +' fold  ' +str(curr_fold) + '  End: test' + '-' * 30) 
                #save model
                if test_acc_fold>best_acc:
                    best_epoch=epoch
                    best_acc=test_acc_fold
                    torch.save(model.state_dict(),checkpoint_dir+'/'+'{}_{}_best_model.ckpt'.format(subject,curr_fold))

            train_used_time_fold = time.time() - train_start_time
            print('Train time: ', train_used_time_fold)

            #no-valid
            #test
            print('-' * 30 + str(subject)+' fold ' +str(curr_fold) + '  Begin: test' + '-' * 30)
            
            test_start_time = time.time()
            with torch.no_grad():
                test_loss=0
                correct=[]
                num_of_true=0
                model.load_state_dict(torch.load(checkpoint_dir+'/'+'{}_{}_best_model.ckpt'.format(subject,curr_fold)))
                model.eval()
                for i,data in enumerate(Testloader,0):
                    data=data.to(DEVICE,non_blocking=True)
                    label=torch.as_tensor(data.y,dtype=torch.long).to(DEVICE)
                    output=model(data.x)
                    test_loss+=criterion(output,label).item()
                    _, predicted = torch.max(output.cpu().data, 1)
                    num_of_true+= (predicted == label.cpu()).sum().item()
                test_loss /= (i+1)
                test_acc_fold=num_of_true/len(Testloader.dataset)
                # y_pred = eval_model.predict(x_test, batch_size=100)  # batch_size = 100
                test_used_time_fold = time.time() - test_start_time
                print('(' + time.asctime(time.localtime(time.time())) + ') subject:',subject,'Test loss:',test_loss,'Test acc:', test_acc_fold, 'Test time: ',test_used_time_fold )
                print('-' * 30 +str(subject) +' fold  ' +str(curr_fold) + '  End: test' + '-' * 30)
                test_accuracy_allfold.append(test_acc_fold)
                #save results
        Acc=pd.DataFrame(test_accuracy_allfold)
        Acc.to_csv(result_dir+'/'+'{}_fold_acc.csv'.format(subject),mode='a',encoding="gbk",header=None,index=False)
        acc.append(stats.mean(test_accuracy_allfold))    
        print(f"10-fold acc:{stats.mean(test_accuracy_allfold)}({stats.stdev(test_accuracy_allfold)})")
    print(f"acc:{stats.mean(acc)}({stats.stdev(acc)})") 