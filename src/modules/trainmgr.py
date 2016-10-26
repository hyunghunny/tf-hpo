
class TrainingManager:
    
    def __init__(self, dataset, params_list):
        self.dataset = dataset
        self.params_list = params_list
        self.num_epochs = 1
        self.logger = None
        self.predictor = None        
        self.dev_type = 'cpu'
        self.num_devs = 1
        self.do_validation = False
        self.pickle_file = PARAMS_LIST_PICKLE_FILE
        
    def setPickleFile(self, pickle_file):
        self.pickle_file = pickle_file
    
    def setTrainingDevices(self, dev_type, num_devs):
        if dev_type is 'cpu':
            self.dev_type = dev_type
        elif dev_type is 'gpu':
            self.dev_type = dev_type
            
        self.num_devs = num_devs
        
    def setEpochs(self, num_epochs):
        self.num_epochs = num_epochs
        
    def setHyperparamTemplate(self, template_file):
        self.template_file = template_file
        
    def setValidationProcess(do_val):
        self.do_validation = do_val
    
    def train(self, params, process_index = 0):
        eval_device_id = '/' + self.dev_type + ':' + str(process_index)
        train_device_id = '/' + self.dev_type + ':' + str(process_index)

        return model.learn(self.dataset, \
                           params,\
                           train_dev = train_device_id, \
                           eval_dev = eval_device_id,\
                           do_validation = self.do_validation, \
                           epochs = self.num_epochs,\
                           logger = self.logger,\
                           predictor = self.predictor)        
    
    def runConcurrent(self, num_processes):
        try:
            working_params_list = []
            while len(params_list) > 0:
                processes = []
                for p in range(num_processes):
                    if len(params_list) is 0:
                        break
                    else:
                        params = params_list.pop(0) # for FIFO
                        working_params_list.append(params)

                        #processes.append(Process(target=train_model, args=(p, dataset, params, epochs, logger, predictor)))
                        processes.append(Process(target=self.train, args=(params, p)))

                # start processes at the same time
                for k in range(len(processes)):
                    processes[k].start()
                # wait until processes done
                for j in range(len(processes)):
                    processes[j].join()
                # XXX: to prepare shutdown
                self.saveRemains(params_list)

        except:
            # save undone params list to pickle file
            remains_list = params_list + working_params_list
            self.saveRemains(remains_list)
            sys.exit(-1)
        
    def saveRemains(params_list):        
        print(str(len(params_list)) + " params remained to learn")
        try:
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(params_list, f)
                print(self.pickle_file + " saved properly")
        except:
            e = sys.exc_info()
            traceback.print_exc()
            print(e) 