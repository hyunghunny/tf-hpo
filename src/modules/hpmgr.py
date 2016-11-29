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

class HPVManager:
    def __init__(self, ini_path, ini_dir='./config/'):
        # for setting timezone
        os.environ['TZ'] = 'Asia/Seoul'
        time.tzset()
        
        self.ini_path = ini_path
        self.ini_dir = ini_dir
        self.read(ini_path)
        
    def read(self, ini_file):
        try:
            parser = ConfigParser.ConfigParser()
            parser.read(ini_file)
        except:
            e = sys.exc_info()
            print("Configuration file error: " + str(e))
            traceback.print_exc()
            
        self.parser = parser
        self.ini_path = ini_file
    
    def getPath(self):
        return os.path.abspath(self.ini_path)
    
    def setExecutionInfo(self, exec_dict):
        for key in exec_dict:
            self.parser.set('Execution', key, exec_dict[key])

    def save(self):
        try:
            cfg_file = open(self.ini_path, 'w')
            sections = self.parser.sections()            
            self.parser.set('Execution', 'created', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            self.parser.write(cfg_file)
            cfg_file.close()            
        
        except:
            e = sys.exc_info()
            print("Save error: " + str(e))
            traceback.print_exc()        
    
    def saveAs(self, new_ini_file=None, prefix="HPV_"):
        if new_ini_file is None:
            date_str = str(int(time.time() * 10000))
            new_ini_file = prefix + date_str + '.ini'
        
        new_ini_file = self.ini_dir + new_ini_file    
        try:
            cfg_file = open(new_ini_file, 'w')
            sections = self.parser.sections()            
            self.parser.set('Execution', 'created', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            self.parser.write(cfg_file)
            cfg_file.close()
            return os.path.abspath(new_ini_file)
        
        except:
            e = sys.exc_info()
            print("Save as error: " + str(e))
            traceback.print_exc()
            
    def validate(self, template_file="HPV_001.ini"):
        try:
            validator = ConfigParser.ConfigParser()
            validator.read(self.ini_dir + template_file)
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
                
                dict1[option] = value
                dict1[option.upper()] = value
                if dict1[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("get section map: exception on %s" % option)
                dict1[option] = None
                dict1[option.upper()] = None
        return dict1
    
    def getOptions(self, section):
        return self.parser.sections()
    
    def getOption(self, section, option, type=None):
        try:
            DebugPrint(section + ":" + option)
            if type == 'bool':
                value = self.parser.getboolean(section, option)
            elif type == 'int':
                value = self.parser.getint(section, option)
            elif type == 'float':
                value = self.parser.getfloat(section, option)                
            else:
                DebugPrint("fallback here")
                value = self.parser.get(section, option)
                DebugPrint(value)
        except:
            print("get option : exception on %s:" % option)
            value = None        
        
        return value

    def setOption(self, section, option, value):
        try:
            #print ('try to set ' + str(value) + ' to ' + str(option) + ' in ' + str(section) + ':' + str(self.ini_path))
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
                
                
                