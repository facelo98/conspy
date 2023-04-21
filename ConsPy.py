import numpy as np
import pandas as pd
import string
from itertools import combinations
from scipy.spatial.distance import cityblock
from itertools import chain
from collections import namedtuple
import warnings
import timeit

warnings.filterwarnings("ignor

class utils:

    "This class contains all the main metrics, object and dependencies to calculate the heuristics algorithms in this package"

    def __init__(self, ranking_matrix: pd.DataFrame = None, ordering_matrix: pd.DataFrame = None, ranks_frequency: pd.DataFrame = None, 
                item_names: list = None, full_ranking: bool = False, get = False):

        if not (type(ranking_matrix) is type(None)) & (type(ordering_matrix) is type(None)):

            self.ranks_frequency = ranks_frequency
            self.rank_types = 'full' if full_ranking is True else 'weak'
            self.item_names = list(ranking_matrix.columns) if item_names is None else item_names
            self.items = ranking_matrix.shape[1] if ranking_matrix is not None else ordering_matrix.shape[1]
            self.judges = ranking_matrix.shape[0] if ranking_matrix is not None else ordering_matrix.shape[0]
            self.ranking_matrix = ranking_matrix if ranking_matrix is not None else self._to_ranking_(ordering_matrix, self.item_names)
            self.ordering_matrix = ordering_matrix if ordering_matrix is not None else self._to_ordering_(ranking_matrix, self.item_names, self.judges, self.items)
            self.combined_Input_matrix = self.combined_input_matrix(x = ranking_matrix, wk = ranks_frequency)

            if get:
                output = heuristics().QUICK(x = ranking_matrix, wk = ranks_frequency, full = full_ranking)
                self.consensus = output[0]
                self.tau = output[1]

    
    @classmethod    
    def _to_ordering_(self, ranking_matrix, item_names, judges, items):

        """
        Transform a ranking matrix into an ordering matrix.

        :ranking_matrix: the ranking matrix
        :return: The ordering dataframe
        """

        ordering_matrix = pd.DataFrame(np.zeros((judges, items)))

        for i in range(judges):
            ordering_matrix.columns = np.sort(ranking_matrix.iloc[i, :])
            for j in range(items):

                if type(ordering_matrix.at[i, ranking_matrix.iloc[i, j]]) != pd.Series:
                    ordering_matrix.at[i, ranking_matrix.iloc[i, j]] = item_names[j]
                else:
                    ties = list(item_names[ranking_matrix.iloc[i, :] == ranking_matrix.iloc[i, j]])

                    if len(ties) == 2:
                        t = [*list(['<' + ties[0]]), *list([ties[len(ties) - 1] + '>'])]
                    else:
                        g = ''
                        for w in range(1, len(ties) - 1):
                            g = g + ties[w]
                        t = [*['<' + ties[0]], *[g], *[ties[w + 1] + '>']]

                    ordering_matrix.at[i, ranking_matrix.iloc[i, j]] = t

        ordering_matrix.columns = range(1, ranking_matrix.shape[1] + 1)
        
        return ordering_matrix


    @classmethod
    def _to_ranking_(self, ordering_matrix, item_names):

        """
        Given an ordering, it is transformed into a ranking

        :param ordering_matrix: The ordering dataframe
        :return: The ranking matrix
        """

        ranking_matrix = pd.DataFrame(np.zeros(ordering_matrix.shape))
        ranking_matrix.columns = item_names

        for i in range(ranking_matrix.shape[0]):
            tied = 0
            exclude = - 1

            for j in range(ranking_matrix.shape[1]):

                if sum(j == pd.Series(exclude)) >= 1:
                    continue

                if sum(ordering_matrix.iloc[i, j] == ranking_matrix.columns) != 0:
                    ranking_matrix.at[i, ordering_matrix.iloc[i, j]] = str(list((ordering_matrix.columns[ordering_matrix.iloc[i, :] == ordering_matrix.iloc[i, j]]) - tied)).strip(
                        '[').strip(']')

                else:
                    ranking_matrix.at[i, ordering_matrix.iloc[i, j][1:len(ordering_matrix.iloc[i, j])]] = str(
                        list((ordering_matrix.columns[ordering_matrix.iloc[i, :] == ordering_matrix.iloc[i, j]]) - tied)).strip('[').strip(']')
                    k = 1

                    while sum(ordering_matrix.iloc[i, j + k] == ranking_matrix.columns) != 0:
                        ranking_matrix.at[i, ordering_matrix.iloc[i, j + k]] = str(list((ordering_matrix.columns[ordering_matrix.iloc[i, :] == ordering_matrix.iloc[i, j]]) - tied)).strip(
                            '[').strip(']')
                        k = k + 1

                        if (j + k) > (ordering_matrix.shape[1] - 1):
                            k = k - 1
                            break

                    ranking_matrix.at[i, ordering_matrix.iloc[i, j + k][0:(len(ordering_matrix.iloc[i, j + k]) - 1)]] = str(
                        list((ordering_matrix.columns[ordering_matrix.iloc[i, :] == ordering_matrix.iloc[i, j]]) - tied)).strip('[').strip(']')
                    tied = tied + k
                    exclude = range(j, j + k + 1)

        for i in range(ranking_matrix.shape[0]):
            for j in range(ranking_matrix.shape[1]):
                ranking_matrix.iloc[i, j] = np.int(ranking_matrix.iloc[i, j])

        return ranking_matrix


    @classmethod
    def kendall_scorematrix(self, x: pd.DataFrame):

        """
        Compute the design matrix to compute kemeny distance

        :param x: A N by M pd.DataFrame, in which there are N judges and M objects to be judged.
        :return: Design matrix
        """

        m = x.shape[1]
        n = x.shape[0]

        ind = pd.DataFrame(combinations(range(1, m + 1), 2))
        kd = pd.DataFrame(np.zeros([n, int(m * (m - 1) / 2)]))

        for j in range(len(ind.index)):
            kd.iloc[:, j] = \
                list(np.transpose(np.sign(x.iloc[:, (ind.iloc[j, 0] - 1)] - x.iloc[:, (ind.iloc[j, 1] - 1)]) * (-1)))

        return kd


    @classmethod
    def kemeny_distance(self, x:pd.DataFrame, y:pd.DataFrame = None):

        """
        Compute the Kemeny distance of a data matrix containing preference rankings, or compute the kemeny distance between
        two (matrices containing) rankings.

        :param x: A N by M pd.dataframe, in which there are N judges and M objects to be judged. Each row is a
                ranking of the objects which are represented by the columns. If there is only X as input,
                the output is a square distance matrix.
        :param y: A row vector, or a n by m data matrix in which there are n judges and the same M objects as
                X to be judged.
        :return: If there is only X as input, distance = square distance matrix. If there is also Y as input, distance =
                matrix with N rows and n columns.
        """

        if type(x) != pd.DataFrame:
            x = pd.DataFrame(x)

        n = len(x.index)

        if isinstance(y, type(None)):

            v = np.matrix(self.kendall_scorematrix(x))
            distance = np.zeros([n, n])

            for i in range(n):
                for j in range(n):
                    if distance[i, j] == 0:
                        distance[i, j] = distance[j, i]  = cityblock(v[i, :], v[j, :])
                    else:
                        pass

        else:
            if type(y) != pd.DataFrame:
                y = pd.DataFrame(y)

            r = len(y.index)
            _x = self.kendall_scorematrix(x)
            _y = self.kendall_scorematrix(y)
            distance = pd.DataFrame(np.zeros((n, r)))

            for j in range(r):
                for i in range(n):
                    distance.iloc[i, j] = np.sum(abs(_x.iloc[i, :] - _y.iloc[j, :]))

        distance = pd.DataFrame(distance)

        return distance


    @classmethod
    def tau_coefficient(self, x:pd.DataFrame, y:pd.DataFrame = None):

        """
        Compute Tau extension rank correlation coefficient (TauX) defined by Emond and Mason (2002)

        :param x: A M by N  pd.dataframe, in which there are N judges and M objects to be judged. Each row
                is a ranking of the objects which are represented by the columns. 
        :param y: A row vector, or a n by M data matrix in which there are n judges and the same M objects
                as X to be judged.
        :return: Tau extension rank correlation coefficient
        """

        if type(x) != pd.DataFrame:
            x = pd.DataFrame(x)

        n = len(x.columns)
        maxd = n * (n - 1)

        if isinstance(y, type(None)):
            d = self.kemeny_distance(x)
            tau = 1 - (2 * d / maxd)
        else:
            if type(y) != pd.DataFrame:
                y = pd.DataFrame(y)
            d = self.kemeny_distance(x, y)
            tau = 1 - (2 * d / maxd)

        return tau


    @classmethod
    def kemeny_scorematrix(self, x:pd.DataFrame):

        """
        Score matrix of rank data according to Kemeny (1962)

        :param x: a ranking (a pd.dataframe, or better a np.matrix with one row and M columns)
        :return: the M by M score matrix
        """
        itemnames = x.columns
        c = x.shape[1]
        sm = pd.DataFrame(np.zeros((c, c)), index = itemnames, columns = itemnames)

        for j in range(c):
            ind = list(chain(range(j), range(j + 1, c)))
            sm.iloc[j, ind] = -(np.sign(x.iloc[0, j] - x.iloc[0, ind]))

        sm = sm.astype(int)
        
        return sm


    @classmethod
    def scorematrix(self, x:pd.DataFrame, get:bool = True):

        """
        Given a ranking, it computes the score matrix as defined by Emond and Mason (2002)

        :param x: a ranking (a pd.dataframe, or better a np.matrix with one row and M columns)
        :return: the M by M score matrix
        """

        itemnames = x.columns
        c = x.shape[1]
        
        x = np.matrix(x)
        
        sm = np.zeros((c, c))

        for j in range(c):
            ind = np.setdiff1d(range(c), j)
            diffs = np.sign(x[0, j] - x[0, ind])
            sm[j, ind] = diffs
            
        idn = np.isnan(sm)
        sm = ((sm <= 0) * 2 - 1)
        np.fill_diagonal(sm, 0)
        sm[idn] = 0
        sm = pd.DataFrame(sm, index=itemnames, columns=itemnames)
        
        if get:
            sm = pd.DataFrame(sm, index=itemnames, columns=itemnames)
        
        return sm


    @classmethod
    def combined_input_matrix(self, x:pd.DataFrame, wk:pd.DataFrame = None):

        """
        Compute the Combined input matrix of a data set as defined by Emond and Mason (2002)

        :param x: A data matrix N by M, in which there are N judges and M objects to be judged. Each row is a ranking of
                the objects which are represented by the columns. Alternatively X can contain the rankings observed only
                once. In this case the argument Wk must be used
        :param wk: Optional, the frequency of each ranking in the data
        :return: The M by M combined input matrix
        """

        names = x.columns
        ci = np.zeros((x.shape[1], x.shape[1]))

        if isinstance(wk, type(None)):
            for i in range(x.shape[0]):
                ci += self.scorematrix(x.iloc[[i]], get=False)
                
        else:
            for i in range(x.shape[0]):
                ci += self.scorematrix(x.iloc[[i]], get=False) * int(wk.iloc[i])
                
        ci = pd.DataFrame(ci, index=names, columns=names)
        
        return ci


    @classmethod
    def stirling2(self, n:int, k:int):

        """
        Stirling numbers of the second kind.

        Denote the number of ways to partition a set of n objects into k non-empty subsets

        :param n: Number of the objects
        :param k: Number of the non-empty subsets (buckets)
        :return: S: The stirling number of the second kind SM: A matrix showing, for each k (on the columns) in how many
                    ways the n objects (on the rows) can be partitioned
        """
        Stirling = namedtuple('Stirling', ('s', 'sm'))

        if k == 0:

            if n == 0:
                s = 1

            else:
                s = 0
            sm = np.zeros(n, n)

        else:
            sm = np.asmatrix(pd.DataFrame(np.zeros((n, k))).replace(0, None))
            sm[:, 0] = 1
            np.fill_diagonal(sm, 1)

            for i in range(1, n):
                crit = min(i, k)

                if crit >= 1:

                    for j in range(1, crit):
                        if pd.isna(sm[(i - 1), j]):
                            sm[i, j] = sm[(i - 1), (j - 1)]

                        else:
                            sm[i, j] = sm[(i - 1), (j - 1)] + (j + 1) * sm[(i - 1), j]

            s = sm[n - 1, k - 1]
        
        return Stirling(s=s, sm=sm)


    @classmethod
    def reordering(self, x:pd.DataFrame):

        """
        Given a ranking of M objects it reduces it in "natural" form

        :param x: A ranking, or a ranking data matrix
        :return: A ranking in natural form
        """

        if x.shape[0] == 1:
            g = x
            ox = np.argsort(x)
            sx = np.sort(x)
            sx = sx - np.min(sx) + 1
            dc = np.vstack([0, np.transpose(np.diff(sx))])

            for i in range(x.shape[1] - 1):

                if dc[i + 1, 0] >= 1:
                    sx[0, i + 1] = sx[0, i] + 1

                else:
                    sx[0, i + 1] = sx[0, i]
            g.iloc[0, ox] = sx

        else:
            g = x
            for j in range(x.shape[0]):
                ox = np.argsort(x.iloc[[j]])
                sx = np.sort(x.iloc[[j]])
                sx = sx - np.min(sx) + 1
                dc = np.vstack([0, np.transpose(np.diff(sx))])

                for i in range(x.shape[1] - 1):
                    if dc[i + 1, 0] >= 1:
                        sx[0, i + 1] = sx[0, i] + 1
                    else:
                        sx[0, i + 1] = sx[0, i]

                g.iloc[j, ox] = sx

        return g


    @classmethod
    def to_ordering(self, x:pd.DataFrame, items:list = False, item_names:list = None, itemtype:str = "N"):

        """
        Given a ranking, it is transformed into an ordering.

        :param x: A n by c ranking dataframe
        :param items: To set true to place new item names.
        :param item_names: The items to be placed into the ordering matrix.
        :param itemtype: To be used only if items is not set. The default value is "N", the columns names are used
                        as items. Instead, using value "L", the first c small letters are placed as items.
        :return: The ordering dataframe
        """

        r = x.shape[0]
        c = x.shape[1]
        out = pd.DataFrame(np.zeros((r, c)))

        if pd.isna(items) and itemtype == "L":
            items = list(string.ascii_uppercase)[0:c]
            x.columns = items

        elif items:
            x.columns = list(item_names)

        col_names = x.columns

        for i in range(r):
            out.columns = np.sort(x.iloc[i, :])
            for j in range(c):

                if type(out.at[i, x.iloc[i, j]]) != pd.Series:
                    out.at[i, x.iloc[i, j]] = col_names[j]
                else:
                    ties = list(x.columns[x.iloc[i, :] == x.iloc[i, j]])

                    if len(ties) == 2:
                        t = [*list(['<' + ties[0]]), *list([ties[len(ties) - 1] + '>'])]
                    else:
                        g = ''
                        for w in range(1, len(ties) - 1):
                            g = g + ties[w]
                        t = [*['<' + ties[0]], *[g], *[ties[w + 1] + '>']]

                    out.at[i, x.iloc[i, j]] = t

        out.columns = range(1, x.shape[1] + 1)

        return out


    @classmethod
    def to_ranking(self, x:pd.DataFrame, item_names:list):

        """
        Given an ordering, it is transformed into a ranking

        :param x: An ordering dataframe.
        :param item_names: Names of the ranked items.
        :return: The ranking matrix
        """

        order = pd.DataFrame(np.zeros(x.shape))
        order.columns = item_names

        for i in range(x.shape[0]):
            tied = 0
            exclude = - 1

            for j in range(x.shape[1]):

                if sum(j == pd.Series(exclude)) >= 1:
                    continue

                if sum(x.iloc[i, j] == order.columns) != 0:
                    order.at[i, x.iloc[i, j]] = str(list((x.columns[x.iloc[i, :] == x.iloc[i, j]]) - tied)).strip(
                        '[').strip(']')

                else:
                    order.at[i, x.iloc[i, j][1:len(x.iloc[i, j])]] = str(
                        list((x.columns[x.iloc[i, :] == x.iloc[i, j]]) - tied)).strip('[').strip(']')
                    k = 1

                    while sum(x.iloc[i, j + k] == order.columns) != 0:
                        order.at[i, x.iloc[i, j + k]] = str(list((x.columns[x.iloc[i, :] == x.iloc[i, j]]) - tied)).strip(
                            '[').strip(']')
                        k = k + 1

                        if (j + k) > (x.shape[1] - 1):
                            k = k - 1
                            break

                    order.at[i, x.iloc[i, j + k][0:(len(x.iloc[i, j + k]) - 1)]] = str(
                        list((x.columns[x.iloc[i, :] == x.iloc[i, j]]) - tied)).strip('[').strip(']')
                    tied = tied + k
                    exclude = range(j, j + k + 1)

        for i in range(order.shape[0]):
            for j in range(order.shape[1]):
                order.iloc[i, j] = np.int(order.iloc[i, j])

        return order


    @classmethod
    def combincost(self, ranking:pd.DataFrame, c:pd.DataFrame, m:int):

        combin = namedtuple('combincost', ('tp', 'cp'))
    
        n = ranking.shape[1]
        s = self.scorematrix(ranking)

        maxdist = n * (n - 1)
        t = np.sum(np.sum(c * s)) / (m * (n * (n - 1)))
        cc = (maxdist / 2) * (1 - t) * m

        return combin(tp=t, cp=cc)


    @classmethod
    def penalty(self, ccr:pd.DataFrame, c:pd.DataFrame, order:pd.DataFrame):

        """
        Determination of penalties for the branch and bound algorithm.

        :param ccr: Candidate consensus ranking
        :param c: Combined input matrix of candidate consensus ranking
        :param order: Ordered vector of candidate consensus ranking
        :return: Penalties of branches for the candidate consensus ranking
        """

        if type(order) != pd.DataFrame:
            order = pd.DataFrame(order)

        i = order.iloc[0, order.shape[1] - 1]
        ds = pd.DataFrame(np.transpose(np.zeros((1, order.shape[1] - 1))))
        tpenalty = pd.DataFrame(np.transpose(np.zeros((1, order.shape[1] - 1))))

        for k in range(order.shape[1] - 1):
            j = order.iloc[0, k]
            ds.iloc[k, 0] = np.sign(ccr.iloc[0, i] - ccr.iloc[0, j])

            if ds.iloc[k, 0] == 1:
                if (np.sign(c.iloc[i, j]) == 1) & (np.sign(c.iloc[j, i]) == -1):
                    tpenalty.iloc[k, 0] = c.iloc[i, j] - c.iloc[j, i]
                elif (np.sign(c.iloc[i, j]) == -1) & (np.sign(c.iloc[j, i]) == 1):
                    tpenalty.iloc[k, 0] = 0
                else:
                    tpenalty.iloc[k, 0] = c.iloc[i, j]

            elif ds.iloc[k, 0] == -1:
                if (np.sign(c.iloc[i, j]) == 1) & (np.sign(c.iloc[j, i]) == -1):
                    tpenalty.iloc[k, 0] = 0
                elif (np.sign(c.iloc[i, j]) == -1) & (np.sign(c.iloc[j, i]) == 1):
                    tpenalty.iloc[k, 0] = c.iloc[j, i] - c.iloc[i, j]
                else:
                    tpenalty.iloc[k, 0] = c.iloc[j, i]

            elif ds.iloc[k, 0] == 0:
                if (np.sign(c.iloc[i, j]) == 1) & (np.sign(c.iloc[j, i]) == -1):
                    tpenalty.iloc[k, 0] = -c.iloc[j, i]
                elif (np.sign(c.iloc[i, j]) == -1) & (np.sign(c.iloc[j, i]) == 1):
                    tpenalty.iloc[k, 0] = -c.iloc[i, j]
                else:
                    tpenalty.iloc[k, 0] = 0
        tpenalty = np.sum(tpenalty)

        return tpenalty


    @classmethod
    def create_data(self, rows, columns, ties):

        dataset = pd.DataFrame(np.zeros((rows, columns)))
        k = 0
        while k <= round(rows*0.2, ndigits=0):
            cons = list(np.random.choice(range(1, columns + 1 - ties), columns - ties, replace=False)) + list(np.random.choice(range(1, columns + 1 - ties), ties, replace=False))
            dataset.iloc[[k]] = cons
            k += 1
            if k >= round(rows*0.2, ndigits=0):
                while k <= round(rows*0.35, ndigits=0):
                    cons = list(np.random.choice(range(1, columns + 1 - ties), columns - ties, replace=False)) + list(np.random.choice(range(1, columns + 1 - ties), ties, replace=False))
                    dataset.iloc[[k]] = cons
                    k += 1
                    if k <= round(rows*0.35, ndigits=0):
                        while k <= round(rows*0.45, ndigits=0):
                            cons = list(np.random.choice(range(1, columns + 1 - ties), columns - ties, replace=False)) + list(np.random.choice(range(1, columns + 1 - ties), ties, replace=False))
                            dataset.iloc[[k]] = cons
                            k += 1
                            if k >= round(rows*0.45, ndigits=0):
                                while k <= round(rows*0.5, ndigits=0):
                                    cons = list(np.random.choice(range(1, columns + 1 - ties), columns - ties, replace=False)) + list(np.random.choice(range(1, columns + 1 - ties), ties, replace=False))
                                    dataset.iloc[[k]] = cons
                                    k += 1
                                    if k >= round(rows*0.5, ndigits=0):
                                        while k <= round(rows*0.55, ndigits=0):
                                            cons = list(np.random.choice(range(1, columns + 1 - ties), columns - ties, replace=False)) + list(np.random.choice(range(1, columns + 1 - ties), ties, replace=False))
                                            dataset.iloc[[k]] = cons
                                            k += 1
                                            if k >= round(rows*0.55, ndigits=0):
                                                while k < rows:
                                                    dataset.iloc[[k]] = list(np.random.choice(range(1, columns + 1 - ties), columns - ties, replace=False)) + list(np.random.choice(range(1, columns + 1 - ties), ties, replace=False))
                                                    k += 1
                dataset = dataset.astype(int)
                
                nomi = pd.DataFrame(combinations(string.ascii_uppercase, 2))
                nomi[1] = nomi[0] + nomi[1]
                nomi = nomi[0].append(nomi[1])
                nomi = nomi.drop_duplicates().values
                dataset.columns = nomi[:columns]

        return dataset


    @classmethod
    def reorderingBB(self, r:pd.DataFrame):

        r = r + 1
        k = r.shape[1]
        neworder = pd.DataFrame(np.argsort(r))
        indexing = pd.DataFrame(np.zeros((1, r.shape[1] - 1)))

        for j in reversed(range(k - 1)):
            indexing.iloc[0, j] = r.iloc[0, neworder.iloc[0, j + 1]] - r.iloc[0, neworder.iloc[0, j]]

        if (indexing.shape[1] - np.count_nonzero(indexing)) > 0:
            k = 0
            while k <= (indexing.shape[1] - 1):
                if indexing.iloc[0, k] == 0:
                    r.iloc[0, neworder.iloc[0, k + 1]] = r.iloc[0, neworder.iloc[0, k]]
                elif indexing.iloc[0, k] > 0:
                    r.iloc[0, neworder.iloc[0, k + 1]] = r.iloc[0, neworder.iloc[0, k]] + 2
                k = k + 1
        else:
            k = 0
            while k <= (indexing.shape[1] - 1):
                r.iloc[0, neworder.iloc[0, k + 1]] = r.iloc[0, neworder.iloc[0, k]] + 2
                k = k + 1
        return r


    @classmethod
    def findbranches(self, r:pd.DataFrame, order:pd.DataFrame, b, full:bool = False):

        """

        :param r: Ranking
        :param order: Sorted ranking
        :param b: Number series
        :param full: True if the ranking is full
        :return: Branches of the candidate consensus ranking
        """
        kr = pd.DataFrame(np.transpose(r.iloc[0, order.iloc[0,b]]))
        kr = kr.iloc[range(kr.size - 1), 0]
        
        mo = np.max(kr)
        mi = np.min(kr)
        aa = 1
        ko = 1
        kr = kr.reset_index(drop=True)
        kr[kr.shape[0]] = mo + 1
        ccr = pd.DataFrame(np.zeros(r.shape))
        ccr.columns = r.columns
        w = 0
        
        while ko == 1:
            ccr = ccr.append(r)
            if aa == 1:
                ccr = ccr.iloc[1:, :]
                
            if full:
                r.iloc[0, order.iloc[0, b[len(b) - 1]]] = r.iloc[0, order.iloc[0, b[len(b) - 1]]] - 2
            else:
                r.iloc[0, order.iloc[0, b[len(b) - 1]]] = r.iloc[0, order.iloc[0, b[len(b) - 1]]] - 1

            
            if (mi - r.iloc[0, order.iloc[0, b[len(b) - 1]]]) > 1:
                ko = 0

            aa = aa + 1

        return ccr


    @classmethod
    def rep(self, x, times):

        """
        :param x: A dataframe to repeat
        :param times: Number of repeat
        :return: The dataframe generated
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame([x])
        global v
        a = 1
        while a <= times:
            if a == 1:
                v = x.append(x, ignore_index=True)
            else:
                v = x.append(v, ignore_index=True)
            a = a + 1
        return pd.DataFrame(v)


    @classmethod
    def which_is_na(self, x):

        """
        :param x: A column vector
        :return: Index containing NA in x
        """

        b = pd.DataFrame(x.apply(pd.isna))
        c = np.zeros((np.sum(b.values == True), 1))
        e = 0
        for i in range(b.shape[0]):
            if b.iloc[i, 0] == True:
                c[e, 0] = b.index[i]
                e = e + 1
        return c[:, 0]


    @classmethod
    def branches(self, brR:pd.DataFrame, c:pd.DataFrame, b, po, order:pd.DataFrame, pb, full:bool = False):

        _branches = namedtuple('_branches', ('r', 'pbr'))
        candidate = self.findbranches(brR, order, b, full=full).dropna(axis='columns')
        pb = self.rep(x=pb, times=candidate.shape[0] - 1)
        crr = pd.DataFrame(np.zeros(candidate.shape))
        addpenalty = pd.DataFrame(np.zeros((candidate.shape[0], 1)))
        qr = pd.DataFrame(np.zeros(candidate.shape))

        for k in range(candidate.shape[0]):
            crr.iloc[k, :] = candidate.iloc[k, :]
            if not isinstance(order, pd.DataFrame):    
                order = pd.DataFrame(order)
            addpenalty.iloc[k, :] = self.penalty(ccr=candidate.iloc[[k]], c=c, order=order.iloc[0, b])
            if (pb.iloc[k] + addpenalty.iloc[k]).sum() > po.sum():
                crr.iloc[k, :] = None
                addpenalty.iloc[k, :] = None
            qr.iloc[k, :] = crr.iloc[k, :]

        pbr = pd.DataFrame(np.zeros(addpenalty.shape))

        for j in range(addpenalty.shape[0]):
            if isinstance(addpenalty.iloc[j, 0], type(None)):
                pbr.iloc[j, 0] = None
            else:
                pbr.iloc[j, 0] = addpenalty.iloc[j, 0] + pb.iloc[j, 0]

        idp = self.which_is_na(pbr)

        if len(idp) == 0:
            r = qr

        elif len(idp) == qr.shape[0]:
            pbr = None
            pb = None
            r = None

        else:
            pbr = pbr.drop(idp)
            r = qr.drop(idp)
            if r.shape[0] == 0:
                r = qr.drop(idp)
        return _branches(r=r, pbr=pbr)


    @classmethod
    def mutate_D5(self, c, wk, full):

        consensus = heuristics().QUICK(x = c, wk = wk, full = full, ps = False)
        return consensus


    @classmethod
    def breakties(self, x:pd.DataFrame):

        """
        Break all the ties in the ordering.

        :param x: Ordering matrix with ties
        :return: Full ordering matrix
        """
        names = list(x.columns)
        x = self.to_ordering(x)
        x[x.columns] = x.apply(lambda t: t.str.strip('<'))
        x[x.columns] = x.apply(lambda t: t.str.strip('>'))
        x = self.to_ranking(x, item_names=names)

        return x


    @classmethod
    def bb_consensus(self, rr:pd.DataFrame, c:pd.DataFrame, full:bool = False, ps:bool = False):

        _bbconsensus = namedtuple('_bbconsensus', ('Consensus', 'Penalty'))
        cr = rr.copy()
        s = self.scorematrix(rr)
        po = sum(np.sum(abs(c))) - sum(np.sum(c * s))
        n = rr.shape[1]
        a = pd.DataFrame(n - np.sort(rr))
        order = pd.DataFrame(np.argsort(-rr))
        r = rr.copy()
        addpenalty = pd.DataFrame(np.zeros((n, 1)))
        alert = False
        for k in range(1, a.shape[1]):
            if k > 1 :
                if candidate.shape[0] == 2:
                    alert = True
                else:
                    alert = False
                
            b = range(k + 1)
            r = self.reorderingBB(r)
            kr = r.iloc[0, order.iloc[0, b]]
            kr = kr.iloc[range(kr.shape[0] - 1)].reset_index(drop=True)
            mo = np.max(kr)
            mi = np.min(kr)
            aa = 0
            ko = 1
            kr[kr.shape[0]] = mo + 1
            r.iloc[0, order.iloc[0, b]] = kr
            candidate = pd.DataFrame(np.zeros(rr.shape))
            pb = [0]
            candidate.columns = r.columns

            while ko == 1:

                candidate = candidate.append(r)

                if aa == 0:
                    candidate = candidate.iloc[range(1, candidate.shape[0]), :]
                
                s_b = self.scorematrix(candidate.iloc[[aa]])
                s_b.columns = s.columns
                s_b.index = s.index
                pb = pb + [sum(np.sum(abs(c))) - sum(np.sum(c * s_b))]
                
                if aa == 0:
                    pb = pb[1:len(pb)]

                if pb[aa] == 0:
                    cr = r
                    po = 0
                    pc = 0
                    break

                pc = 1
                
                if full:
                    r.iloc[0, order.iloc[0, b[len(b) - 1]]] = r.iloc[0, order.iloc[0, b[len(b) - 1]]] - 2
                else:
                    r.iloc[0, order.iloc[0, b[len(b) - 1]]] = r.iloc[0, order.iloc[0, b[len(b) - 1]]] - 1

                if np.array(mi - r.iloc[0, order.iloc[0, b[len(b) - 1]]]) > 1:
                    ko = 0

                aa = aa + 1

            if ps:
                print('Evaluated ' + str(candidate.shape[0]) + ' branches')

            if pc == 0:
                break

            minp = np.min(pb)
            posp = np.where(pb == minp)

            if minp <= po:
                po = minp.copy()
                cr = candidate.iloc[posp[0]]
                r = cr.copy()
                addpenalty.iloc[[k]] = self.penalty(r, c, order.iloc[0, b])
            else:
                r = cr.copy()
                addpenalty.iloc[[k]] = self.penalty(r, c, order.iloc[0, b])
            break

            
        if pc == 0:
            po = 0
            addpenalty = 0
        else:
            poo = np.sum(addpenalty)

        s_c = self.scorematrix(cr)
        po = np.sum(addpenalty)
        
        return _bbconsensus(Consensus=cr, Penalty=po)


    @classmethod
    def bb_consensus2(self, rr:pd.DataFrame, c:pd.DataFrame, po, ps:bool = False, full:bool = False):

        cr = rr.copy()
        a = np.sort(rr)
        order = pd.DataFrame(np.argsort(rr))
        r = self.reorderingBB(rr)
        br_r = r.copy()
        br_p = [0]
        wci = 1
        lamda = 1
        nobj = rr.shape[1]
        while wci == 1:
            if ps:
                print('round ' + str(lamda))

            for k in range(1, nobj):
                b_ = br_r.shape[0]
                b = list(range(k + 1))
                for nb in range(b_):
                    br_r.iloc[[nb]] = self.reorderingBB(br_r.iloc[[nb]])
                    if isinstance(br_p, int):
                        br_p = list(br_p)
                    rpbr = self.branches(brR=br_r.iloc[[nb]], c=c, b=b, po=po, order=order, pb=br_p[nb], full=full)
                    r = rpbr[0].iloc[[0]]
                    pbr = rpbr[1].iloc[[0]]
                    if isinstance(rpbr[0], type(None)):
                        continue
                    else:
                        if nb == 0:
                            kr_r = r.copy()
                            kr_p = pbr.copy()
                        else:
                            kr_r = kr_r.append(r)
                            kr_p = kr_p.append(pbr)
                
                if isinstance(r, type(None)):
                    if (nb == (b_ - 1)) & (nb != 0):
                        br_r = kr_r.copy()
                        br_p = kr_p.copy()
                        kr_r = None
                        kr_p = None
                    else:
                        continue
                else:
                    br_r = kr_r.copy()
                    br_p = kr_p.copy()
                    kr_r = None
                    kr_p = None
                if ps:
                    print('evaluating ' + str(b_) + ' branches')
            minp = np.min(br_p)
            ssp = np.where(br_p == minp)
            penmin = po - minp

            if np.array(penmin) == 0:
                if br_r.shape[0] == 1:
                    cr = br_r.copy()
                else:
                    cr = br_r.iloc[[ssp]]
                wci = 0

            else:
                po = minp.copy()
                wci = 1
                lamda = lamda + 1
                nrr = br_r.iloc[ssp[0]]
                br_r = nrr.copy()
                br_p = 0
                a = np.sort(br_r)
                order = np.argsort(br_r)

        return cr


    @classmethod
    def findconsensus_BB(self, c:pd.DataFrame, full:bool = False):

        x = pd.DataFrame(np.zeros((1, c.shape[1]))) + 1
        n = x.shape[1]
        idx = pd.DataFrame(combinations(range(n), 2))

        for j in range(len(idx)):
            if (np.sign(c.iloc[idx.iloc[j, 0], idx.iloc[j, 1]]) == 1) & \
                    (np.sign(c.iloc[idx.iloc[j, 1], idx.iloc[j, 0]]) == - 1):
                x.iloc[0, idx.iloc[j, 0]] = x.iloc[0, idx.iloc[j, 0]] + 1
            elif (np.sign(c.iloc[idx.iloc[j, 0], idx.iloc[j, 1]]) == - 1) & \
                    (np.sign(c.iloc[idx.iloc[j, 1], idx.iloc[j, 0]]) == 1):
                x.iloc[0, idx.iloc[j, 1]] = x.iloc[0, idx.iloc[j, 1]] + 1
            elif (np.sign(c.iloc[idx.iloc[j, 0], idx.iloc[j, 1]]) == - 1) & \
                    (np.sign(c.iloc[idx.iloc[j, 1], idx.iloc[j, 0]]) == - 1):
                x.iloc[0, idx.iloc[j, 0]] = None
            elif (np.sign(c.iloc[idx.iloc[j, 0], idx.iloc[j, 1]]) == 1) & \
                    (np.sign(c.iloc[idx.iloc[j, 1], idx.iloc[j, 0]]) == 1):
                x.iloc[0, idx.iloc[j, 0]] = x.iloc[0, idx.iloc[j, 0]] + 1
                x.iloc[0, idx.iloc[j, 1]] = x.iloc[0, idx.iloc[j, 1]] + 1

        x = (n + 1) - x
        x.columns = list(c.columns)

        if full:
            x = self.breakties(x)

        return x


    @classmethod
    def mutaterand1(self, x:pd.DataFrame, ff, i):

        d = x.shape[0]
        a = np.random.choice(range(d), d, replace=False)
        for j in [0, 1, 2]:
            if a[j] == i:
                a[j] = a[3]
        r1 = a[0]
        r2 = a[1]
        r3 = a[2]

        v = x.iloc[[r1]] + ff * (x.iloc[r2, :] - x.iloc[r3, :])

        return v


    @classmethod
    def crossover(self, x:pd.DataFrame, v, cr):

        d = x.shape[1]
        u = pd.DataFrame(np.zeros((1, d)))

        for i in range(d):
            if np.random.uniform(0, 1) > cr:
                u.iloc[0, i] = v.iloc[0, i]
            else:
                u.iloc[0, i] = x.iloc[0, i]
        return u


    @classmethod
    def childtie(self, r:pd.DataFrame):

        o = r.rank(method='average', axis=1)
        o = self.reordering(o)
        return o


    @classmethod
    def childclosint(self, r:pd.DataFrame):

        d = r.shape[1]
        x = r.round(decimals=0)

        for i in range(d):

            if (np.array(x.iloc[:, i]) > d) | (np.array(x.iloc[:, i]) < 1):
                r = np.random.choice(d, replace=False)
                x.iloc[:, i] = r[0]

            c = np.setdiff1d(np.union1d(r, list(range(1, d + 1))), np.intersect1d(r, list(range(1, d + 1))))

            if len(c) == 0:
                x = x

            else:
                u = np.transpose(x.T.drop_duplicates())
                id = np.where(x.T.duplicated() == False)
                ix = np.setdiff1d(np.union1d(id, list(range(d))), np.intersect1d(id, list(range(d))))
                x.iloc[:, list(ix)] = c[list(np.random.choice(range(len(ix)), len(ix), replace=False))]

        return x


class heuristics:
    
    "This class contains the functions to solve the consensus ranking problem through the heuristics algorithms: BB, FAST, QUICK and DECOR. \n they can handle full, weak or incomplete rankings"

    @classmethod
    def bb_weak(self, x:pd.DataFrame, wk:pd.DataFrame = None, ps:bool = True):

        "Compute the branch and bound algorithm to solve the consensus ranking problem using weak or incomplete rankings"

        bb_weak = namedtuple('Branch_and_Bound_algorithm', ('Consensus', 'Tau'))
        m = x.shape[0]
        n = x.shape[1]
        
        if m == 1:
            consensus = x
            tau_x = 1

        else:
            if not isinstance(wk, type(None)):
                c = utils().combined_input_matrix(x, wk)
            else:
                c = utils().combined_input_matrix(x, None)


            if np.sum(c.values == 0) == np.product(c.shape):
                Consensus = 'Combined input matrix contains only zeroes: any ranking in the reference universe is a median ranking'
                return Consensus

            elif np.sum(np.sum(np.sign(c + np.identity(c.shape[0])))) == np.product(c.shape):
                Consensus = 'Combined Input Matrix contains only positive values: the median ranking is the all-tie solution'
                return Consensus

            else:

                r = utils().findconsensus_BB(c=c)

                cons1 = utils().bb_consensus(rr=r, c=c, full=False, ps=False)
                consensus1 = cons1[0]
                po = cons1[1]
                consensus = utils().bb_consensus2(rr=consensus1, c=c, po=po, ps=ps, full=False)

                if consensus.shape[0] == 1:
                    s = utils().scorematrix(consensus)
                    s.columns = c.columns
                    s.index = c. index

                    if not isinstance(wk, type(None)):
                        tau_x = np.sum((c * s).values) / (np.sum(wk) * (n * (n - 1)))
                    else:
                        tau_x = np.sum((c * s).values) / (m * (n * (n - 1)))

                else:
                    tau_x = pd.DataFrame(np.zeros((consensus.shape[0], 1)))

                    for k in range(consensus.shape[0]):
                        s = utils().scorematrix(consensus.iloc[[k]])
                        s.columns = c.columns
                        s.index = c.index

                        if not isinstance(wk, type(None)):
                            tau_x.iloc[[k]] = np.sum((c * s).values) / (np.sum(wk) * (n * (n - 1)))
                        else:
                            tau_x.iloc[[k]] = np.sum((c * s).values) / (m * (n * (n - 1)))

            consensus.columns = x.columns
            
        return bb_weak(Consensus=utils().reordering(consensus), Tau=tau_x)


    @classmethod
    def bb_full(self, x:pd.DataFrame, wk:pd.DataFrame = None, ps:bool = True):
        
        "Compute the branch and bound algorithm to solve the consensus ranking problem using full rankings"
        
        bb_full = namedtuple('Branch_and_Bound_algorithm', ('Consensus', 'Tau'))
        m = x.shape[0]
        n = x.shape[1]
        
        if m == 1:
            consensus = x
            tau_x = 1

        else:
            if not isinstance(wk, type(None)):
                c = utils().combined_input_matrix(x, wk)
            else:
                c = utils().combined_input_matrix(x)

            if np.sum(c.values == 0) == np.product(c.shape):
                Consensus = 'Combined input matrix contains only zeroes: any ranking in the reference universe is a median ranking'
                return Consensus
            else:
                r = utils().findconsensus_BB(c=c, full=True)
                cons1 = utils().bb_consensus(rr=r, c=c, full=True, ps=False)
                consensus1 = cons1[0]
                po = cons1[1]
                consensus = utils().bb_consensus2(rr=consensus1, c=c, po=po, ps=ps, full=True)

                if consensus.shape[0] == 1:
                    s = utils().scorematrix(consensus)
                    s.columns = c.columns
                    s.index = c. index

                    if not isinstance(wk, type(None)):
                        tau_x = np.sum((c * s).values) / (np.sum(wk) * (n * (n - 1)))
                    else:
                        tau_x = np.sum((c * s).values) / (m * (n * (n - 1)))
                else:
                    tau_x = pd.DataFrame(np.zeros((consensus.shape[0], 1)))

                    for k in range(consensus.shape[0]):
                        s = utils().scorematrix(consensus.iloc[[k]])
                        s.columns = c.columns
                        s.index = c.index

                        if not isinstance(wk, type(None)):
                            tau_x.iloc[[k]] = np.sum((c * s).values) / (np.sum(wk) * (n * (n - 1)))
                        else:
                            tau_x.iloc[[k]] = np.sum((c * s).values) / (m * (n * (n - 1)))

            consensus.columns = x.columns

        return bb_full(Consensus=utils().reordering(consensus), Tau=tau_x)


    def FAST(self, x:pd.DataFrame, wk:pd.DataFrame = None, maxiter:int = 100, full:bool = False, ps:bool = False):

        "Compute the FAST algorithm to solve the consensus ranking problem using full, weak or incomplete rankings"

        fast = namedtuple('FAST_algorithm', ('Consensus', 'Tau'))
        m = x.shape[0]
        n = x.shape[1]
        
        if m == 1:
            cr = x.copy()
            tau = 1
        else:
            if not isinstance(wk, type(None)):
                c = utils().combined_input_matrix(x, wk)
            else:
                c = utils().combined_input_matrix(x)

            if np.sum(c.values == 0) == np.product(c.shape):
                Consensus = 'Combined input matrix contains only zeroes: any ranking in the reference universe is a median ranking'
                return Consensus

            elif np.sum(np.sum(np.sign(c + np.identity(c.shape[0])))) == np.product(c.shape):
                Consensus = 'Combined Input Matrix contains only positive values: the median ranking is the all-tie solution'
                return Consensus

            else:
                cr = pd.DataFrame(np.zeros((maxiter, x.shape[1])))
                for i in range(maxiter):
                    if full:
                        r = pd.DataFrame(np.matrix(np.random.choice(range(1, n + 1), n, replace=False)))
                    else:
                        if (i + 1) % 2 == 0:
                            r = pd.DataFrame(np.matrix(np.random.choice(range(1, n + 1), n, replace=True)))
                        else:
                            r = pd.DataFrame(np.matrix(np.random.choice(range(1, n + 1), n, replace=False)))
                    consensus1 = utils().bb_consensus(rr=r, c=c, full=full)
                    cons = consensus1[0].reset_index(drop=True).iloc[[0]]
                    cr.iloc[[i]] = cons
                    if ps:
                        print('Iteration ' + str(i))

                tau = pd.DataFrame(np.zeros((cr.shape[0], 1)))
                for k in range(cr.shape[0]):
                    s = utils().scorematrix(cr.iloc[[k]])
                    s.columns = c.columns
                    s.index = c.index
                    if not isinstance(wk, type(None)):
                        tau.iloc[[k]] = np.sum((c * s).values) / (np.sum(wk) * (n * (n - 1)))
                    else:
                        tau.iloc[[k]] = np.sum((c * s).values) / (m * (n * (n - 1)))

                cr = utils().reordering(cr)
                cr['TAU'] = tau
                cr = cr[cr['TAU'] == cr['TAU'].max()]
                cr.drop(columns = 'TAU', inplace = True)
                tau = np.max(tau)
                if cr.shape[0] > 1:
                    cr = cr.drop_duplicates()
                if not isinstance(cr.shape[0], type(None)):
                    tau = pd.DataFrame(np.repeat(tau, cr.shape[0]))

        return fast(Consensus=cr, Tau=tau)


    def QUICK(self, x:pd.DataFrame, wk:pd.DataFrame = None, full:bool = False, ps:bool = False):

        "Compute the QUICK algorithm to solve the consensus ranking problem using full, weak or incomplete rankings"

        quick = namedtuple('QUICK_algorithm', ('Consensus', 'Tau'))
        callfull = full
        callps = ps

        m = x.shape[0]
        n = x.shape[1]

        if m==1:
            consensus = x.copy()
            tau = 1
        else:
            if not isinstance(wk, type(None)):
                c = utils().combined_input_matrix(x, wk)
            else:
                c = utils().combined_input_matrix(x)

            if np.sum(c.values == 0) == np.product(c.shape):
                Consensus = 'Combined input matrix contains only zeroes: any ranking in the reference universe is a median ranking'
                return Consensus
            elif np.sum(np.sum(np.sign(c + np.identity(c.shape[0])))) == np.product(c.shape):
                Consensus = 'Combined Input Matrix contains only positive values: the median ranking is the all-tie solution'
                return Consensus
            else:
                r = utils().findconsensus_BB(c=c, full=callfull)
                r1 = (n - 1) - r
                consensusA = utils().bb_consensus(rr=r, c=c, full=callfull, ps=callps)[0]
                consensusB = utils().bb_consensus(rr=consensusA, c=c, full=callfull, ps=callps)[0]
                consensusC = utils().bb_consensus(rr=consensusB, c=c, full=callfull, ps=callps)[0]
                consensusD = utils().bb_consensus(rr=consensusC, c=c, full=callfull, ps=callps)[0]
                consensus = consensusA.append([consensusB, consensusC, consensusD], ignore_index=True)
                consensus = utils().reordering(consensus).drop_duplicates()
                howcons = consensus.shape[0]

                tau = pd.DataFrame(np.zeros((consensus.shape[0], 1)))
                for k in range(consensus.shape[0]):
                    s = utils().scorematrix(consensus.iloc[[k]])
                    if not isinstance(wk, type(None)):
                        tau.iloc[[k]] = np.sum((c * s).values) / (np.sum(wk) * (n * (n - 1)))
                    else:
                        tau.iloc[[k]] = np.sum((c * s).values) / (m * (n * (n - 1)))

                if howcons > 1:
                    nco = np.where(tau == np.max(tau))
                    if len(nco) > 1:
                        consensus = consensus[[nco]]
                        tau = pd.DataFrame(np.repeat(np.max(tau), consensus.shape[0], 0))
                    else:
                        tau = np.max(tau)
                        consensus = consensus.iloc[[nco]]
                tau.columns = ['Tau']

        return quick(Consensus=utils().reordering(consensus), Tau=tau)



    def DECOR(self, c:pd.DataFrame, wk:pd.DataFrame, nj:int, nP:int = 15, l:int = 50, ff:int = 0.4, cr:int = 0.9, full:bool = False):

        "Compute the DECOR algorithm to solve the consensus ranking problem using full, weak or incomplete rankings"

        decor = namedtuple('DECOR', ('Consensus', 'Tau'))
        n = c.shape[1]
        population = pd.DataFrame(np.zeros((nP - 1, n)))
        population.columns = c.columns
        if isinstance(type(c), type(None)):
            for k in range(nP - 1):
                population.iloc[[k]] = np.random.choice(range(1, n + 1), n, replace=False)

            
            population = population.append(utils().findconsensus_BB(c=c, full=full))
            costs = pd.DataFrame(np.zeros((nP, 1)))
            taos = costs.copy()

            for i in range(nP):
                cota = utils().combincost(ranking=population.iloc[[i]], c=c, m=nj)
                costs.iloc[[i]] = cota[1]
                taos.iloc[[i]] = cota[0]

            bestc = np.min(costs)
            bestind = np.where(costs == bestc)
            bestT = pd.DataFrame(np.max(taos))
            besti = population.iloc[bestind[0], :]
            g = 1
            no_gain = 0

            while no_gain < l:
                for i in range(nP):
                    evolution = utils().mutaterand1(x=population, ff=ff, i=i)
                    evolution = utils().crossover(x=population.iloc[[i]], v=evolution, cr=cr)
                    if full:
                        evolution = np.argsort(evolution)
                    else:
                        evolution = utils().childtie(evolution)

                    cotan = utils().combincost(ranking=evolution, c=c, m=nj)
                    cost_new = cotan[1]
                    ta_new = cotan[0]
                    if cost_new < costs.iloc[i, 0]:
                        population.iloc[[i]] = evolution
                        costs.iloc[i, 0] = cost_new
                        taos.iloc[[i]] = ta_new

                bestco = np.min(costs)
                bestc = pd.DataFrame(bestc).append(bestco, ignore_index=True)
                bestind = np.where(costs == bestco)
                bestTa = pd.DataFrame(np.max(taos))
                bestT = bestT.append(bestTa, ignore_index=True)
                bestin = population.iloc[bestind[0]]
                besti = besti.append(bestin)

                if np.array(bestc.iloc[[g]]) == np.array(bestc.iloc[[g-1]]):
                    no_gain += 1
                else:
                    no_gain = 0

                g += 1

            indexes = pd.DataFrame(np.where(bestc == np.min(bestc)))

            if full:
                if indexes.shape[1] == 1:
                    bests = utils().childclosint(besti.iloc[indexes])
                else:
                    bests = pd.DataFrame(np.zeros((indexes.shape[1], n)))
                    for j in range(indexes.shape[1]):
                        bests.iloc[[j]]= utils().childclosint(besti.iloc[[indexes.iloc[0, j]]]) 
                        
            else:
                if indexes.shape[1] == 1:
                    bests = utils().reordering(besti.iloc[indexes])
                else:
                    for j in range(indexes.shape[1]):
                        bests = utils().reordering(besti.iloc[[indexes.iloc[0, j]]])

            avgTau = bestT.iloc[[indexes.iloc[0, j]]]
            consR = bests.drop_duplicates()
            tau = avgTau[[0]]
        else:
            _best_ = utils().mutate_D5(c, wk, full)

        return decor(Consensus=_best_[0], Tau=_best_[1]) 


    def conspy(self, x:pd.DataFrame, wk:pd.DataFrame=None, full:bool = False, ps:bool = False, algorithm:str = 'BB', maxiter:float = 100, nP:float = 15, gl:float = 100, ff:float = 0.4, cr:float = 0.9):

        start_processing = timeit.default_timer()

        if not isinstance(x, pd.DataFrame):
            return "<Type error>: {} Conspy works only with pandas dataframe.".format(type(x))
        else:
            if algorithm == 'BB':  
                if full:
                    if x.shape[1] > 50:
                        answer_1 = input(prompt='It is not recommended to use the Branch and Bound algorithm with more than 50 objects. \nType Y if you want to change algorithm or N to procede with Branch and Bound \n')
                        if answer_1.capitalize() == 'Y':
                            answer_2 = input(prompt='Type Q for quick, F for fast or D for decor')
                            if answer_2.capitalize() == 'Q':
                                return self.QUICK(x, wk, full, ps)
                            elif answer_2.capitalize() == 'F':
                                return self.FAST(x, wk, maxiter, full, ps)
                            elif answer_2.capitalize() == 'D':
                                return self.DECOR(x, nP, l = gl, ff = ff, cr = cr)
                            else:
                                return 'Not valid input'
                        elif answer_1.capitalize() == 'N':
                            consensus = self.bb_full(x, wk, ps)
                        else:
                            return 'Not valid input'
                    else:
                        consensus = self.bb_full(x, wk, ps)
                else:
                    if x.shape[1] > 50:
                        answer_1 = input(prompt='It is not recommended to use the Branch and Bound algorithm with more than 50 objects. \nType Y if you want to change algorithm or N to procede with Branch and Bound \n')
                        if answer_1.capitalize() == 'Y':
                            answer_2 = input(prompt='Type Q for quick, F for fast or D for DEcoR')
                            if answer_2.capitalize() == 'Q':
                                consensus = self.QUICK(x, wk, full, ps)
                            elif answer_2.capitalize() == 'F':
                                consensus = self.FAST(x, wk, maxiter, full, ps)
                            elif answer_2.capitalize() == 'D':
                                consensus = self.DECOR(x, nj = x.shape[1], nP = nP, l = gl, ff = ff, cr = cr)
                            else:
                                return 'Not valid input'
                        elif answer_1.capitalize() == 'N':
                            consensus = self.bb_weak(x, wk, ps)
                        else:
                            return 'Not valid input'
                    else:
                        consensus = self.bb_weak(x, wk, ps)
            elif algorithm == 'FAST':
                consensus = self.FAST(x, wk, maxiter, full, ps)
            elif algorithm == 'QUICK':
                consensus = self.QUICK(x, wk, full, ps)
            elif algorithm == 'DECOR':
                consensus = self.DECOR(x, wk, nj = x.shape[1], nP = nP, l = gl, ff = ff, cr = cr)
            
            print('Processing time: {}'.format(timeit.default_timer() - start_processing))

        return consensus
























