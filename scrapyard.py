def disdance_matric():
    a = [[0, 0, 0, 0, 0, 0, 306, 294, 0, 0, 173, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 114, 121, 0, 0, 0, 0, 0, 160, 0, 0, 0],
         [0, 0, 0, 0, 37, 0, 0, 211, 0, 94, 0, 0, 0, 313, 0, 0, 0],
         [0, 0, 0, 0, 34, 0, 0, 0, 0, 39, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 37, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 354, 0, 190, 396, 84, 0, 0, 224, 0, 0],
         [306, 114, 0, 0, 0, 0, 0, 158, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [294, 121, 211, 0, 0, 354, 158, 0, 0, 0, 267, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 79, 0],
         [0, 0, 94, 39, 0, 190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [173, 0, 0, 0, 0, 396, 0, 267, 0, 0, 0, 0, 0, 0, 282, 0, 0],
         [0, 0, 0, 0, 0, 84, 0, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 0, 156],
         [0, 160, 313, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 224, 0, 0, 0, 0, 282, 0, 169, 0, 0, 210, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 210, 0, 92],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 0, 0, 92, 0]]

    print('[')
    for ii, i in enumerate(a):
        string = ''
        for jj, j in enumerate(i):
            # string += '\t'
            if ii == jj:
                string += '0'
            elif j == 0:
                string += 'Inf'
            else:
                string += str(j)
            string += ','
        string += ';'
        print(string)
    print(']')


def configs_generation():
    data_rates = [i for i in range(100, 650, 50)][::-1]
    bit_per_symbol = [i for i in range(2, 7)][::-1]
    fec_percentage = [i for i in [0, 0.15, 0.27]]
    baud_rate_range = [32, 72]

    # dual-polarization
    all_configs = [[i, j, k] for i in data_rates for j in bit_per_symbol for k in fec_percentage]
    available_configs = []
    for config_i in all_configs:
        baud_rate_i = (config_i[0] / (1 - config_i[2])) / (config_i[1] * 2)
        config_i += [baud_rate_i]
        if baud_rate_range[0] < baud_rate_i < baud_rate_range[1]:
            available_configs += [config_i]
    available_configs.sort(key=lambda x: (-x[0], x[3]))
    return available_configs


class Test:
    def __init__(self):
        self.dict = [[1, 2, 3, 4], [5, 6, 7, 8]]

    def fun(self):
        for i, x in enumerate(self.dict):
            x[2] += 3
            print(x)
            self.dict = [[1, 2, 3, 4], [5, 6, 7, 8]]


test = Test()
test.fun()
print(123)
