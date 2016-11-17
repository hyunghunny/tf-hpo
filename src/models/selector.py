
class ModelSelector:
    def __init__(self, data):
        self.dataset = data
        
    def select(self, num_conv, num_fc):
        model = None
        if num_conv == 1:
            if num_fc == 1:
                from models.cnn_1c1f import CNN1C1F
                model = CNN1C1F(self.dataset)
            elif num_fc == 2:
                from models.cnn_1c2f import CNN1C2F
                model = CNN1C2F(self.dataset)
            elif num_fc == 3:
                from models.cnn_1c3f import CNN1C3F
                model = CNN1C3F(self.dataset)                
        elif num_conv == 2:        
            if num_fc == 1:
                from models.cnn_2c1f import CNN2C1F
                model = CNN2C1F(self.dataset)
            elif num_fc == 2:
                from models.cnn_2c2f import CNN2C2F
                model = CNN2C2F(self.dataset)
            elif num_fc == 3:
                from models.cnn_2c3f import CNN2C3F
                model = CNN2C3F(self.dataset)
        elif num_conv == 3:
            if num_fc == 1:
                from models.cnn_3c1f import CNN3C1F
                model = CNN3C1F(self.dataset)
            elif num_fc == 2:
                from models.cnn_3c2f import CNN3C2F
                model = CNN3C2F(self.dataset)
            elif num_fc == 3:
                from models.cnn_3c3f import CNN3C3F
                model = CNN3C3F(self.dataset)
        return model