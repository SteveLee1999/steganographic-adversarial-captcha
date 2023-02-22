'''
https://github.com/daniellerch/stegolab/blob/master/codes/STC.py
'''
import numpy as np
B = 8  # STC: number of copies of H_hat in H
STCODE = [71, 109]

class STC:
    # b是H所包含H^的个数
    def __init__(self):
        # 二进制字符串，从2开始取是因为bin返回值含有前缀'0b'
        n_bits = len(bin(np.max(STCODE))[2:])
        M = []
        for d in STCODE:
            # ljust = 左对齐填充0至指定长度,n_bits是最长者的长度
            M.append(np.array([int(x) for x in list(bin(d)[2:].ljust(n_bits, '0'))]))
        H_hat = np.array(M).T
        n, m = H_hat.shape
        H = np.zeros((n + B - 1, m * B))
        for i in range(B):
            H[i:i + n, m * i:m * (i + 1)] = H_hat
        self.code_b = B
        self.code_n = m * B
        self.code_l = n_bits
        self.code_h = np.tile(STCODE, B)
        self.code_shift = np.tile([0] * (m - 1) + [1], B)

    def _dual_viterbi(self, x, w, m):
        C = np.zeros((2 ** self.code_l, self.code_n))
        costs = np.infty * np.ones((2 ** self.code_l, 1))
        costs[0] = 0
        paths = np.zeros((2 ** self.code_l, self.code_n))
        m_id = 0  # message bit id
        y = np.zeros(x.shape)

        # Run forward
        for i in range(self.code_n):
            costs_old = costs.copy()
            hi = self.code_h[i]
            ji = 0
            for j in range(2 ** self.code_l):
                c1 = costs_old[ji] + x[i] * w[i]
                c2 = costs_old[(ji ^ hi)] + (1 - x[i]) * w[i]
                if c1 < c2:
                    costs[j] = c1
                    paths[j, i] = ji  # store index of the previous path
                else:
                    costs[j] = c2
                    paths[j, i] = ji ^ hi  # store index of the previous path
                ji = ji + 1

            for j in range(self.code_shift[i]):
                tail = np.infty * np.ones((2 ** (self.code_l - 1), 1))
                if m[m_id] == 0:
                    costs = np.vstack((costs[::2], tail))
                else:
                    costs = np.vstack((costs[1::2], tail))

                m_id = m_id + 1

            C[:, i] = costs[:, 0]

        ind = np.argmin(costs)
        min_cost = costs[ind, 0]

        m_id -= 1

        for i in range(self.code_n - 1, -1, -1):
            for j in range(self.code_shift[i]):
                ind = 2 * ind + m[m_id, 0]  # invert the shift in syndrome trellis
                m_id = m_id - 1

            y[i] = paths[ind, i] != ind
            ind = int(paths[ind, i])

        return y.astype('uint8'), min_cost, paths

    def _calc_syndrome(self, x):
        m = np.zeros((np.sum(self.code_shift), 1))
        m_id = 0
        tmp = 0
        for i in range(self.code_n):
            hi = self.code_h[i]
            if x[i] == 1:
                tmp = hi ^ tmp
            for j in range(self.code_shift[i]):
                m[m_id] = tmp % 2
                # tmp = tmp >> pipeline
                tmp //= 2
                m_id += 1
        return m.astype('uint8')

    def _bytes_to_bits(self, m):
        bits = []
        for b in m:
            for i in range(8):
                bits.append((b >> i) & 1)
        return bits

    def _bits_to_bytes(self, m):
        enc = bytearray()
        idx = 0
        bitidx = 0
        bitval = 0
        for b in m:
            if bitidx == 8:
                enc.append(bitval)
                bitidx = 0
                bitval = 0
            bitval |= b << bitidx
            bitidx += 1
        if bitidx == 8:
            enc.append(bitval)
        return bytes(enc)

    def embed(self, cover, costs, message):
        """
        :param cover: list of pixel values, cover image
        :param costs: list of cost values, cost function of pixels in cover
        :return: steganographic image
        """
        x = np.array(cover)
        w = np.array(costs)
        # 将message中每个字符展开为8bit(1byte)
        message_bits = np.array(self._bytes_to_bits(message))
        i = 0
        j = 0
        y = x.copy()
        while True:
            # np.newaxis用于创建新维度
            # eg. a.shape = (2, 3), a[:,:,np.newaxis].shape = (2, 3, pipeline)
            # 这里x_chunk.shape = (self.code_n, pipeline)
            x_chunk = x[i:i + self.code_n][:, np.newaxis] % 2
            w_chunk = w[i:i + self.code_n][:, np.newaxis]
            m_chunk = message_bits[j:j + self.code_b][:, np.newaxis]
            y_chunk, min_cost, _ = self._dual_viterbi(x_chunk, w_chunk, m_chunk)
            # if x_chunk[i] != y_chunk[i], then idx = True; or idx = False
            idx = x_chunk[:, 0] != y_chunk[:, 0]
            # if idx[k] == True, then y[i+k]++;
            y[i:i + self.code_n][idx] += 1
            i += self.code_n
            j += self.code_b
            if i + self.code_n > len(x) or j + self.code_b > len(message_bits):
                break
        return y

    def extract(self, stego):
        y = stego
        message = []
        for i in range(0, len(y), self.code_n):
            y_chunk = y[i:i + self.code_n][:, np.newaxis] % 2
            if len(y_chunk) < self.code_n:
                break
            m_chunk = self._calc_syndrome(y_chunk)
            message += m_chunk[:, 0].tolist()
        message_bytes = self._bits_to_bytes(message)
        return message_bytes


if __name__ == "__main__":
    message = ('a'*100).encode('utf8')
    stcode = [71, 109]  # See Table pipeline for other good syndrome-trellis codes.
    stc = STC(stcode, 8)
    stego = stc.embed(cover, costs, message)
    extracted_message = stc.extract(stego)

