# Hopfield Network

#Imports
import numpy as np
import pandas as pd

class hopfield(object):

    def __init__(self, patterns, noise_percentage, pattern_n_row, pattern_n_column, ib, epochs, neta):
        self.patterns = patterns
        self.noise = 1 - noise_percentage
        self.nrow = pattern_n_row
        self.ncol = pattern_n_column
        self.fmn = len(patterns)
        self.dim = len(self.patterns[0])
        self.ib = ib
        self.epc = epochs
        self.scape = False
        self.n = neta

    def noise_attribution(self, patt):
        self.pattern = patt
        self.randM = np.random.rand(self.nrow, self.ncol)
        self.auxA = self.noise > self.randM
        self.auxB = self.noise < self.randM
        self.randM[self.auxA] = 1
        self.randM[self.auxB] = -1
        self.new_patter = self.pattern.reshape(self.nrow, self.ncol) * self.randM
        return self.new_patter.reshape(self.dim, 1)

    def weights(self):
        self.auxW = 0

        for patt in self.patterns:
            self.auxW += patt * patt.reshape(self.dim, 1)

        a = 1 / self.dim
        b = self.fmn / self.dim

        self.W = (a * self.auxW) - (b * np.zeros((self.dim, self.dim)))

    def run(self):

        self.weights()
        self.it_patt = []
        self.outputs = pd.DataFrame()
        self.noised_img = pd.DataFrame()

        for patt in self.patterns:
            self.v_current = self.noise_attribution(patt)
            self.noised_img = pd.concat((self.noised_img, pd.DataFrame(self.v_current).T))
            self.it = 0
            self.scape = False

            while (self.scape == False):
                self.v_past = self.v_current
                self.u = np.dot(self.W, self.v_past) + self.ib
                self.u -= self.n * self.u
                self.v_current = np.sign(np.tanh(self.u))

                if pd.DataFrame(self.v_current).equals(pd.DataFrame(self.v_past)):
                    self.scape = True

                if (self.it >= self.epc):
                    self.scape = True

                self.it += 1
            self.it_patt.append(self.it)

            self.outputs = pd.concat((self.outputs, pd.DataFrame(self.v_current).T))

def analise():
    perc = []
    acerto = 0

    for elemento, out in zip(livro.values, hp.outputs.values):

        erro = 0

        for i in range(nrow * ncol):
            if elemento[i] != out[i]:
                erro += 1
            else:
                acerto += 1

        perc.append(erro)

    total_perc = np.array(perc) * 100 / (nrow * ncol)
    med_perc = np.mean(total_perc)

    acerto_perc = (acerto * 100) / (hp.ncol * hp.nrow * hp.fmn)

    print('Acerto %: ' + np.str(acerto_perc))
    print('Erro medio %:' + np.str(med_perc))

