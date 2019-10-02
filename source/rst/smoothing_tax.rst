.. _smoothing_tax:

.. include:: /_static/includes/header.raw

.. highlight:: python3


***********************************************
More Finite Markov Chain Tax-Smoothing Examples
***********************************************

.. index::
    single: Tax

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture uses the library:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Import dependent packages.

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    %matplotlib inline

Define the `ConsumptionProblem` class and functions for computing return rates.

.. code-block:: python3

    class ConsumptionProblem:
        """
        The data for a consumption problem, including some default values.
        """

        def __init__(self,
                     β=.96,
                     y=[2, 1.5],
                     b0=3,
                     P=[[.8, .2],
                        [.4, .6]],
                     init=0):
            """
            Parameters
            ----------

            β : discount factor
            y : list containing the two income levels
            b0 : debt in period 0 (= initial state debt level)
            P : 2x2 transition matrix
            init : index of initial state s0
            """
            self.β = β
            self.y = np.asarray(y)
            self.b0 = b0
            self.P = np.asarray(P)
            self.init = init

        def simulate(self, N_simul=150, random_state=1):
            """
            Parameters
            ----------

            N_simul : number of periods for simulation
            random_state : random state for simulating Markov chain
            """
            # For the simulation define a quantecon MC class
            mc = qe.MarkovChain(self.P)
            s_path = mc.simulate(N_simul, init=self.init, random_state=random_state)

            return s_path


    def consumption_complete(cp):
        """
        Computes endogenous values for the complete market case.

        Parameters
        ----------

        cp : instance of ConsumptionProblem

        Returns
        -------

            c_bar : constant consumption
            b : optimal debt in each state

        associated with the price system

            Q = β * P
        """
        β, P, y, b0, init = cp.β, cp.P, cp.y, cp.b0, cp.init   # Unpack

        Q = β * P                               # assumed price system

        # construct matrices of augmented equation system
        n = P.shape[0] + 1

        y_aug = np.empty((n, 1))
        y_aug[0, 0] = y[init] - b0
        y_aug[1:, 0] = y

        Q_aug = np.zeros((n, n))
        Q_aug[0, 1:] = Q[init, :]
        Q_aug[1:, 1:] = Q

        A = np.zeros((n, n))
        A[:, 0] = 1
        A[1:, 1:] = np.eye(n-1)

        x = np.linalg.inv(A - Q_aug) @ y_aug

        c_bar = x[0, 0]
        b = x[1:, 0]

        return c_bar, b


    def consumption_incomplete(cp, s_path):
        """
        Computes endogenous values for the incomplete market case.

        Parameters
        ----------

        cp : instance of ConsumptionProblem
        s_path : the path of states
        """
        β, P, y, b0 = cp.β, cp.P, cp.y, cp.b0  # Unpack

        N_simul = len(s_path)

        # Useful variables
        n = len(y)
        y.shape = (n, 1)
        v = np.linalg.inv(np.eye(n) - β * P) @ y

        # Store consumption and debt path
        b_path, c_path = np.ones(N_simul+1), np.ones(N_simul)
        b_path[0] = b0

        # Optimal decisions from (12) and (13)
        db = ((1 - β) * v - y) / β

        for i, s in enumerate(s_path):
            c_path[i] = (1 - β) * (v - b_path[i] * np.ones((n, 1)))[s, 0]
            b_path[i + 1] = b_path[i] + db[s, 0]

        return c_path, b_path[:-1], y[s_path]

.. code-block:: python3

    def ex_post_gross_return(b, cp):
        """
        calculate the ex post one-period gross return on the portfolio
        of government assets, given b and Q.
        """
        Q = cp.β * cp.P

        values = Q @ b

        n = len(b)
        R = np.zeros((n, n))

        for i in range(n):
            ind = cp.P[i, :] != 0
            R[i, ind] = b[ind] / values[i]

        return R

    def cumulative_return(s_path, R):
        """
        compute cumulative return from holding 1 unit market portfolio
        of government bonds, given some simulated state path.
        """
        T = len(s_path)

        RT_path = np.empty(T)
        RT_path[0] = 1
        RT_path[1:] = np.cumprod([R[s_path[t], s_path[t+1]] for t in range(T-1)])

        return RT_path


Here we give more examples of tax-smoothing models with both complete and incomplete markets in an :math:`N` state Markov setting.

These examples differ in how Markov states are jumping between peace and war.

To wrap the procedure of solving models, relabeling the graph so that we record government *debt* rather than *assets*,
and displaying the results, we define a new class below.

.. code-block:: python3

    class TaxSmoothingExample:
        """
        construct a tax-smoothing example, by relabeling consumption problem class.
        """
        def __init__(self, g, P, b0, states, β=.96,
                     init=0, s_path=None, N_simul=150, random_state=1):

            self.states = states # state names

            # if the path of states is not specified
            if s_path is None:
                self.cp = ConsumptionProblem(β, g, b0, P, init=init)
                self.s_path = self.cp.simulate(N_simul=N_simul, random_state=random_state)
            # if the path of states is specified
            else:
                self.cp = ConsumptionProblem(β, g, b0, P, init=s_path[0])
                self.s_path = s_path

            # solve for complete market case
            self.T_bar, self.b = consumption_complete(self.cp)
            self.debt_value = - (β * P @ self.b).T

            # solve for incomplete market case
            self.T_path, self.asset_path, self.g_path = \
                consumption_incomplete(self.cp, self.s_path)

            # calculate returns on state-contingent debt
            self.R = ex_post_gross_return(self.b, self.cp)
            self.RT_path = cumulative_return(self.s_path, self.R)

        def display(self):

            # plot graphs
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            N = len(self.T_path)

            ax[0].set_title('Tax collection paths')
            ax[0].plot(np.arange(N), self.T_path, label='incomplete market')
            ax[0].plot(np.arange(N), self.T_bar * np.ones(N), label='complete market')
            ax[0].plot(np.arange(N), self.g_path, label='govt expenditures', alpha=.6, ls='--')
            ax[0].legend()
            ax[0].set_xlabel('Periods')

            ax[1].set_title('Government debt paths')
            ax[1].plot(np.arange(N), -self.asset_path, label='incomplete market')
            ax[1].plot(np.arange(N), -self.b[self.s_path], label='complete market')
            ax[1].plot(np.arange(N), self.g_path, label='govt expenditures', ls='--')
            ax[1].plot(np.arange(N), self.debt_value[self.s_path], label="today's value of debts")
            ax[1].legend()
            ax[1].axhline(0, color='k', ls='--')
            ax[1].set_xlabel('Periods')

            ax[2].set_title('Cumulative return path (complete market)')
            ax[2].plot(np.arange(N), self.RT_path, color='b')
            ax[2].set_xlabel('Periods')
            ax[2].set_ylabel('Cumulative return', color='b')

            ax2_ = ax[2].twinx()
            ax2_.plot(np.arange(N), self.g_path, ls='--', color='g')
            ax2_.set_ylabel('Government expenditures', color='g')

            plt.show()

            # plot detailed information
            Q = self.cp.β * self.cp.P

            print(f"P \n {self.cp.P}")
            print(f"Q \n {Q}")
            print(f"Govt expenditures in {', '.join(self.states)} = {self.cp.y.flatten()}")
            print(f"Constant tax collections = {self.T_bar}")
            print(f"Govt debt in {len(self.states)} states = {-self.b}")

            print("")
            print(f"Government tax collections minus debt levels in {', '.join(self.states)}")
            for i in range(len(self.states)):
                TB = self.T_bar + self.b[i]
                print(f"  T+b in {self.states[i]} = {TB}")

            print("")
            print(f"Total government spending in {', '.join(self.states)}")
            for i in range(len(self.states)):
                G = self.cp.y[i, 0] + Q[i, :] @ self.b
                print(f"  {self.states[i]} = {G}")

            print("")
            print("Let's see ex-post and ex-ante returns on Arrow securities \n")

            print(f"Ex-post returns to purchase of Arrow securities:")
            for i in range(len(self.states)):
                for j in range(len(self.states)):
                    if Q[i, j] != 0.:
                        print(f"  π({self.states[j]}|{self.states[i]}) = {1/Q[i, j]}")

            print("")
            exant = 1 / self.cp.β
            print(f"Ex-ante returns to purchase of Arrow securities = {exant}")

            print("")
            print("The Ex-post one-period gross return on the portfolio of government assets")
            print(self.R)

            print("")
            print("The cumulative return earned from holding 1 unit market portfolio of government bonds")
            print(self.RT_path[-1])

Parameters
----------

.. code-block:: python3

    γ = .1
    λ = .1
    ϕ = .1
    θ = .1
    ψ = .1
    g_L = .5
    g_M = .8
    g_H = 1.2
    β = .96

Example 1
---------

This example is designed to produce some stylized versions of tax, debt, and deficit paths followed by the United States during and
after the Civil War and also during and after World War I.

We set the Markov chain to have three states

.. math::
    P =
    \begin{bmatrix}
        1 - \lambda & \lambda  & 0    \cr
        0           & 1 - \phi & \phi \cr
        0           & 0        & 1
    \end{bmatrix}

where the government expenditure vector  :math:`g = \begin{bmatrix} g_L & g_H & g_M \end{bmatrix}` where :math:`g_L < g_M < g_H`.

We set :math:`b_0 = 1` and assume that the initial Markov state is state :math:`1` so that the system starts off in peace.

These parameters have government expenditure beginning at a low level, surging during the war, then decreasing after the war to a level
that exceeds its prewar level.

(This type of  pattern occurred in the US Civil War and World War I experiences.)

.. code-block:: python3

    g_ex1 = [g_L, g_H, g_M]
    P_ex1 = np.array([[1-λ, λ,  0],
                      [0, 1-ϕ,  ϕ],
                      [0,   0,  1]])
    b0_ex1 = 1
    states_ex1 = ['peace', 'war', 'postwar']

.. code-block:: python3

    ts_ex1 = TaxSmoothingExample(g_ex1, P_ex1, b0_ex1, states_ex1, random_state=1)
    ts_ex1.display()

.. code-block:: python3

    # The following shows the use of the wrapper class when a specific state path is given
    # (for Tom)
    s_path = [0, 0, 1, 1, 2]
    ts_s_path = TaxSmoothingExample(g_ex1, P_ex1, b0_ex1, states_ex1, s_path=s_path)
    ts_s_path.display()

Example 2
---------

This example captures a peace followed by a war, eventually followed by a  permanent peace .

Here we set

.. math::
    P =
    \begin{bmatrix}
        1    & 0        & 0      \cr
        0    & 1-\gamma & \gamma \cr
        \phi & 0        & 1-\phi
    \end{bmatrix}

where the government expenditure vector :math:`g = \begin{bmatrix} g_L & g_L & g_H \end{bmatrix}` where :math:`g_L < g_H`.

We assume :math:`b_0 = 1` and that the initial Markov state is state :math:`2` so that the system starts off in a temporary peace.

.. code-block:: python3

    g_ex2 = [g_L, g_L, g_H]
    P_ex2 = np.array([[1,   0,    0],
                      [0, 1-γ,    γ],
                      [ϕ,   0, 1-ϕ]])
    b0_ex2 = 1
    states_ex2 = ['peace', 'temporary peace', 'war']

.. code-block:: python3

    ts_ex2 = TaxSmoothingExample(g_ex2, P_ex2, b0_ex2, states_ex2, init=1, random_state=1)
    ts_ex2.display()

Example 3
---------

This example features a situation in which one of the states is a war state with no hope of peace next period, while another state
is a war state with a positive probability of peace next period.

The Markov chain is:

.. math::
    P =
    \begin{bmatrix}
   		1 - \lambda & \lambda  & 0      & 0         \cr
        0           & 1 - \phi & \phi   & 0         \cr
        0           & 0        & 1-\psi & \psi      \cr
        \theta      & 0        & 0      & 1 - \theta
    \end{bmatrix}

with government expenditure levels for the four states being
:math:`\begin{bmatrix} g_L & g_L & g_H & g_H \end{bmatrix}` where :math:`g_L < g_H`.


We start with :math:`b_0 = 1` and :math:`s_0 = 1`.

.. code-block:: python3

	g_ex3 = [g_L, g_L, g_H, g_H]
	P_ex3 = np.array([[1-λ,  λ,   0,    0],
	                  [0,  1-ϕ,   ϕ,     0],
	                  [0,    0,  1-ψ,    ψ],
	                  [θ,    0,    0,  1-θ ]])
	b0_ex3 = 1
	states_ex3 = ['peace1', 'peace2', 'war1', 'war2']

.. code-block:: python3

	ts_ex3 = TaxSmoothingExample(g_ex3, P_ex3, b0_ex3, states_ex3, random_state=1)
	ts_ex3.display()


Example 4
---------



Here the Markov chain is:

.. math::
	P =
    \begin{bmatrix}
   		1 - \lambda & \lambda  & 0      & 0          & 0      \cr
		0           & 1 - \phi & \phi   & 0          & 0      \cr
        0           & 0        & 1-\psi & \psi       & 0      \cr
        0           & 0        & 0      & 1 - \theta & \theta \cr
        0           & 0        & 0      & 0          & 1
    \end{bmatrix}

with government expenditure levels for the five states being
:math:`\begin{bmatrix} g_L & g_L & g_H & g_H & g_L \end{bmatrix}` where :math:`g_L < g_H`.

We ssume that :math:`b_0 = 1` and :math:`s_0 = 1`.

.. code-block:: python3

	g_ex4 = [g_L, g_L, g_H, g_H, g_L]
	P_ex4 = np.array([[1-λ,  λ,   0,     0,    0],
	                  [0,  1-ϕ,   ϕ,     0,    0],
	                  [0,    0,  1-ψ,    ψ,    0],
	                  [0,    0,    0,   1-θ,   θ],
	                  [0,    0,    0,     0,   1]])
	b0_ex4 = 1
	states_ex4 = ['peace1', 'peace2', 'war1', 'war2', 'permanent peace']

.. code-block:: python3

	ts_ex4 = TaxSmoothingExample(g_ex4, P_ex4, b0_ex4, states_ex4, random_state=1)
	ts_ex4.display()

Example 5
---------

The  example captures a case when  the system follows a deterministic path from peace to war, and back to peace again.

Since there is no randomness, the outcomes in complete markets setting should be the same as in incomplete markets setting.

The Markov chain is:

.. math::
    P =
    \begin{bmatrix}
   		0 & 1 & 0 & 0 & 0 & 0 & 0 \cr
        0 & 0 & 1 & 0 & 0 & 0 & 0 \cr
        0 & 0 & 0 & 1 & 0 & 0 & 0 \cr
        0 & 0 & 0 & 0 & 1 & 0 & 0 \cr
        0 & 0 & 0 & 0 & 0 & 1 & 0 \cr
        0 & 0 & 0 & 0 & 0 & 0 & 1 \cr
        0 & 0 & 0 & 0 & 0 & 0 & 1 \cr
    \end{bmatrix}

with government expenditure levels for the seven states being
:math:`\begin{bmatrix} g_L & g_L & g_H & g_H &  g_H & g_H & g_L \end{bmatrix}` where
:math:`g_L < g_H`. Assume :math:`b_0 = 1` and :math:`s_0 = 1`.

.. code-block:: python3

	g_ex5 = [g_L, g_L, g_H, g_H, g_H, g_H, g_L]
	P_ex5 = np.array([[0, 1, 0, 0, 0, 0, 0],
	                  [0, 0, 1, 0, 0, 0, 0],
	                  [0, 0, 0, 1, 0, 0, 0],
	                  [0, 0, 0, 0, 1, 0, 0],
	                  [0, 0, 0, 0, 0, 1, 0],
	                  [0, 0, 0, 0, 0, 0, 1],
	                  [0, 0, 0, 0, 0, 0, 1]])
	b0_ex5 = 1
	states_ex5 = ['peace1', 'peace2', 'war1', 'war2', 'war3', 'permanent peace']

.. code-block:: python3

	ts_ex5 = TaxSmoothingExample(g_ex5, P_ex5, b0_ex5, states_ex5, N_simul=7, random_state=1)
	ts_ex5.display()
