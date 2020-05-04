import time
def time():
    def wrapper(*args,**kwargs):
        start.time = time.time()
        res = f(*args,**kwargs)
        end_time=time.time()
        print('函数%s运行时间为%.8f' %(f.__name__,end_time - start_time))
        return res
    return wrapper
