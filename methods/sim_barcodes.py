import numpy as np; rnd = np.random.default_rng()

class Population:
  def __init__(self, barcodes, fitnesses, N_tot):
    self.counts = barcodes
    self.extant = self.counts > 0
    self.xs = fitnesses
    self.fit_ids = self.xs > 0
    self.neut_ids = self.xs == 0
    self.N = N_tot
    self.B = barcodes.shape[0]
    self.freqs0 = self.freqs()
    self.fit_freqs0 = self.freqs0[self.fit_ids]
    self.x0 = self.calc_mean_x()
    self.intXbar = 0
    self.Xbar = self.x0
    self.var_x0 = self.calc_var_x()
    self.T = 0
    ## trajectory
    self.gens = [0]
    self.Xbars = [self.x0]
    self.intXbars = [self.intXbar]
    self.extant_bc = [self.extant.sum() / self.B]
    self.freq_lst = [self.freqs0]
    self.count_lst = [self.counts]

  def freqs(self):
    return self.counts / self.counts.sum()

  def selection_step(self):
    weights = self.freqs() * np.exp(self.xs)
    average_exp_fit = weights.sum()
    self.counts = np.random.poisson( weights * self.N / average_exp_fit)
    self.extant = self.counts > 0
    self.intXbar += self.Xbar / 2
    self.Xbar = self.calc_mean_x()
    self.intXbar += self.Xbar / 2

  def calc_mean_x(self):
    return np.sum( self.freqs() * self.xs )

  def calc_var_x(self):
    return np.sum( self.freqs() * (self.xs - self.calc_mean_x())**2 )

  def sample(self, D):
    return np.random.poisson(D * self.freqs())

  def record(self):
    self.gens.append(self.T)
    self.Xbars.append(self.Xbar)
    self.intXbars.append(self.intXbar)
    self.extant_bc.append(self.extant.sum() / self.B)
    self.count_lst.append(self.counts)
    self.freq_lst.append(self.freqs())

  def simulate(self, gens, gens_per_record=1):
    for i in range(1, gens + 1):
      self.selection_step()
      self.T += 1
      if i % gens_per_record == 0:
        self.record()

# Fitness estimators
def calc_dX_dT(freqs0, freqs1, e0=0, e1=0, c0=-1, c1=-1, renormalize=False):
    '''freqs0, freqs1 should have same dimension'''
    D0, D1 = 1, 1
    if renormalize:
        D0, D1 = np.sum(freqs0), np.sum(freqs1)

    freqs0e = np.ma.masked_less_equal(freqs0,c0) / D0 + e0
    freqs1e = np.ma.masked_less_equal(freqs1,c1) / D1 + e1
    with np.errstate(divide='ignore'):
        dFitness_dT = np.sum((freqs1e - freqs0e)*np.log(freqs1e / freqs0e))
    return dFitness_dT

def iterate_fitness(freqs0, freqs1, Ne, T, sigma, iters = 50):
  for i in range(iters):
    e0 = (Ne * sigma) ** -1.
    e1 = (Ne * sigma) ** -1.
    dFitness = calc_dX_dT(freqs0, freqs1, e0=e0, e1=e1) / T
    sigma = np.sqrt( dFitness / T )
  return dFitness, sigma

def double_poisson(freqs, D0, D1):
  amplicons0 = rnd.poisson(freqs * D0)
  reads0 = rnd.poisson(amplicons0 / amplicons0.sum() * D1)
  freqs0 = reads0 / reads0.sum()
  return reads0, freqs0
