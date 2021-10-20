
class Checker:
    def __init__(self, hin):
        super().__init__()
        self.hin = hin
        self.chain = []
    
    def conv(self, kernel, stride=1, pad=0, dilation=1):
        self.chain.append([0, kernel, stride, pad, dilation, 0])
    
    def convT(self, kernel, stride=1, pad=0, dilation=1, outpad=0):
        self.chain.append([1, kernel, stride, pad, dilation, outpad])
    def pool(self, kernel, stride=None, pad=0, dilation=1):
        stride = kernel if stride is None else stride
        self.chain.append([2, kernel, stride, pad, dilation, 0])
    
    def get_shape(self, verbose=False):
        h = self.hin
        for idx, c in enumerate(self.chain):
            i, k, s, p, d, o = c
            if i == 0 or i == 2:
                _h = (h + 2*p - d*(k-1) - 1)//s + 1
            else:
                if o >= s or o >= k:
                    print("[fail] output padding too large")        
                    return
                _h = (h-1)*s - 2*p + d*(k-1) + o + 1
            if verbose:
                print(f"{idx}: {h} -> {_h}")
            h = _h
        
        return h
    
    def check(self, outshape=None):
        outshape = self.hin if outshape is None else outshape
        h = self.get_shape(True)
        if h != outshape:
            print("[fail] check fail")
        else:
            print("[pass] check pass")


check = Checker(224)
check.conv(213, 32, 92)
check.check(7)

exit(0)

check = Checker(1024)
check.conv(3, 2, 2)
check.conv(5, 2, 3)
check.conv(7, 2, 4)
# check.conv(3, 1)
csh = check.get_shape(True)
print("")

check = Checker(1024)
check.conv(2*2*7 + 2*5 + 3 - (2 + 6), 8, 2*2*4 + 2*3 + 2)
# check.conv(2*5 + 3 - 2, 4, 2*3 + 2)
check.check(csh)

print("")

check = Checker(28)
check.conv(3, 2, 1)
check.conv(3, 2, 1)
# check.conv(3, 2)
# check.conv(3, 1)
csh = check.get_shape(True)
print("")

check = Checker(28)
# check.conv(15, 8, 3)
check.conv(7, 4, 3)
# check.conv(2*5 + 3 - 2, 4, 2*3 + 2)
check.check(csh)
