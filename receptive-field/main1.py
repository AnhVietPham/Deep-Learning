import math

# https://rubikscode.net/2021/11/15/receptive-field-arithmetic-for-convolutional-neural-networks/

alex_net = {
    'conv1': [11, 4, 0],
    'pool1': [3, 2, 0],
    'conv2': [5, 1, 2],
    'pool2': [3, 2, 0],
    'conv3': [3, 1, 1],
    'conv4': [3, 1, 1],
    'pool5': [3, 2, 0],
    'fc6-conv': [6, 1, 0],
    'fc7-conv': [1, 1, 0]
}


class ReceptiveFieldCalculator():
    def calculate(self, architecture, input_image_size):
        input_layer = ('input_layer', input_image_size, 1, 1, 0.5)
        self._print_layer_info(input_layer)

        for key in architecture:
            current_layer = self._calculate_layer_info(architecture[key], input_layer, key)
            self._print_layer_info(current_layer)
            input_layer = current_layer

    def _print_layer_info(self, layer):
        print(f'------')
        print(f'{layer[0]}: n = {layer[1]}; r = {layer[2]}; j = {layer[3]}; start = {layer[4]}')
        print(f'------')

    def _calculate_layer_info(self, current_layer, input_layer, layer_name):
        n_in = input_layer[1]
        j_in = input_layer[2]
        r_in = input_layer[3]
        start_in = input_layer[4]

        k = current_layer[0]
        s = current_layer[1]
        p = current_layer[2]

        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        padding = (n_out - 1) * s - n_in + k
        p_right = math.ceil(padding / 2)
        p_left = math.floor(padding / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - p_left) * j_in
        return layer_name, n_out, j_out, r_out, start_out


if __name__ == '__main__':
    calculator = ReceptiveFieldCalculator()
    calculator.calculate(alex_net, 227)
