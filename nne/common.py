import platform

def check_jetson():
    if platform.machine() == 'aarch64':
        return True
    else:
        return False
