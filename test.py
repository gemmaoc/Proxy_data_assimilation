
from load_data import load_cpickle

class config(object):
    def __init__(self,core,proxies,psm):
        self.core = core()
        self.proxies = proxies()
        self.psm = psm()

Cfg = config(v_core,v_proxies,v_psm)

try:
     pre_calib_file = config.psm.linear.pre_calib_datafile
        
     return load_cpickle(pre_calib_file)
