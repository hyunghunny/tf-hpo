import ConfigParser
import os
import sys
import traceback
import time
import datetime

DEBUG = False

def DebugPrint(*args):
    if DEBUG:
        print(args)

class ConfigManager:
    def __init__(self, config_file, config_dir='./config/'):
        # for setting timezone
        os.environ['TZ'] = 'Asia/Seoul'
        time.tzset()
        
        self.config_dir = config_dir
        self.read(config_dir + config_file)
        
    def read(self, config_file):
        try:
            parser = ConfigParser.ConfigParser()
            parser.read(config_file)
        except:
            e = sys.exc_info()
            print("Configuration file error: " + str(e))
            traceback.print_exc()
            
        self.parser = parser
        self.config_file = config_file
    
    def getConfigPath(self):
        return self.config_file
    
    def setExecutionInfo(self, exec_dict):
        for key in exec_dict:
            self.parser.set('Execution', key, exec_dict[key])

    def write(self, new_config_file=None):
        if new_config_file is None:
            date_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
            new_config_file = 'HPV_' + date_str + '.ini'
        new_config_file = self.config_dir + new_config_file    
        try:
            cfg_file = open(new_config_file, 'w')
            sections = self.parser.sections()            
            self.parser.set('Execution', 'created', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            self.parser.write(cfg_file)
            cfg_file.close()
        except:
            e = sys.exc_info()
            print("Configuration file error: " + str(e))
            traceback.print_exc()
            
    def validate(self, template_file="HPV_001.ini"):
        try:
            validator = ConfigParser.ConfigParser()
            validator.read(self.config_dir + template_file)
        except:
            e = sys.exc_info()
            print("Configuration file error: " + str(e))
            traceback.print_exc()        
        
        missing_list = []
        sections = validator.sections()
        for section in sections:
            options = validator.options(section)
            for option in options:
                result = self.getOption(section, option)
                if result is None:
                    missing = '[' + section + '] ' + str(option)                    
                    missing_list.append(missing)
        
        if len(missing_list) is 0:
            return True
        else:
            print ('Missing required hyperparameters: ' + str(missing_list))
            return False        

    def getSectionMap(self, section):       
        options = self.parser.options(section)        
        option_types = self.parser.options("TypeInformation")
        dict1 = {}
        for option in options:
            option = option.lower()
            try:
                option_type = self.parser.get("TypeInformation", option)
            except:
                option_type = 'string'
            try:                
                #DebugPrint(option_type)
                value = self.getOption(section, option, option_type)
                #DebugPrint(type(value))
                option = option.upper()
                dict1[option] = value
                if dict1[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("get section map: exception on %s" % option)
                dict1[option.upper()] = None
        return dict1
    
    def getOptions(self, section):
        return self.parser.sections()
    
    def getOption(self, section, option, type=None):
        try:
            #DebugPrint(type)
            if type == 'bool':
                value = self.parser.getboolean(section, option)
            elif type == 'int':
                value = self.parser.getint(section, option)
            elif type == 'float':
                value = self.parser.getfloat(section, option)                
            else:        
                value = self.parser.get(section, option)
        except:
            print("get option : exception on %s:" % option)
            value = None        
        
        return value

    def setOption(self, section, option, value):
        try:
            #print ('try to set ' + str(value) + ' to ' + option)
            self.parser.set(section, option, str(value))
        except:
            traceback.print_exc()

    def show(self):
        sections = self.parser.sections()
        for section in sections:
            print('[' + section + ']')
            options = self.parser.options(section)
            for option in options:
                result = self.getOption(section, option)
                print(option + "=" + str(result))
                
                
                