
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

POSITIVE = 4.0  # a rating of 4 or higher counts as "liked"


class BaseRecommender:
    name = "base"

    def fit(self, train: pd.DataFrame):
        """train needs the columns userId, movieId, rating."""
        raise NotImplementedError

    def recommend(self, profile: dict, k: int = 10):
        """profile is {movieId: rating}. Returns a list of recommendations."""
        raise NotImplementedError

    def recommend_user(self, user_id: int, k: int = 10):
        """Recommend for a user that is in the training data."""
        return self.recommend(self.user_profiles.get(user_id, {}), k)

    # ---- shared helpers -------------------------------------------------

    def _prepare(self, train: pd.DataFrame):
        """Things every model needs: item index, user profiles, popularity."""
        counts = train.groupby("movieId").size().sort_values(ascending=False)
        self.pop_counts = counts.to_dict()
        self.pop_rank = counts.index.tolist()

        self.item_ids = np.array(sorted(train["movieId"].unique()))
        self.item_index = {int(m): i for i, m in enumerate(self.item_ids)}
        self.user_ids = np.array(sorted(train["userId"].unique()))
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}

        self.user_profiles = {
            int(u): dict(zip(g["movieId"].astype(int), g["rating"].astype(float)))
            for u, g in train.groupby("userId")
        }

    def _liked_matrix(self, train: pd.DataFrame):
        """Binary matrix, users x items, 1 where the rating is >= POSITIVE."""
        liked = train[train["rating"] >= POSITIVE]
        rows = liked["userId"].map(self.user_index).values
        cols = liked["movieId"].map(self.item_index).values
        return csr_matrix(
            (np.ones(len(liked)), (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )

    def _liked_in_index(self, profile: dict):
        """Liked movies of a profile that the model knows about."""
        return [int(m) for m, r in profile.items()
                if r >= POSITIVE and int(m) in self.item_index]

    def _seen_mask(self, profile: dict, scores: np.ndarray):
        """Set the score of already seen movies to -inf so they are never returned."""
        for m in profile:
            i = self.item_index.get(int(m))
            if i is not None:
                scores[i] = -np.inf
        return scores


# ---------------------------------------------------------------------------
# 1. Popularity
# ---------------------------------------------------------------------------

class Popularity(BaseRecommender):
    """Most rated movies in the training data. Same list for everybody, minus
    what the user has already seen. This is the baseline."""

    name = "popularity"

    def fit(self, train):
        self._prepare(train)
        return self

    def recommend(self, profile, k=10):
        seen = {int(m) for m in profile}
        out = []
        for m in self.pop_rank:
            if m not in seen:
                out.append({"movieId": int(m), "score": float(self.pop_counts[m]), "reasons": []})
                if len(out) == k:
                    break
        return out


# ---------------------------------------------------------------------------
# 2. Item-to-item kNN
# ---------------------------------------------------------------------------

class ItemKNN(BaseRecommender):
    """Two movies are similar when the same users liked both. 
    A movie's score is the sum of its similarity to each movie the user liked. 
    Reasons are the liked movies with the largest
    similarity to the recommendation."""

    name = "item_knn"

    def __init__(self, fallback=True):
        self.fallback = fallback  # use popularity if none of the liked movies are known

    def fit(self, train):
        self._prepare(train)
        U = self._liked_matrix(train)          # users x items
        self.M = normalize(U.T.tocsr())        # items x users, rows have length 1
        self._pop = Popularity().fit(train)
        return self

    def recommend(self, profile, k=10):
        used = self._liked_in_index(profile)
        if not used:
            return self._pop.recommend(profile, k) if self.fallback else []

        idx = [self.item_index[m] for m in used]
        S = (self.M @ self.M[idx].T).tocsr()   # items x liked items
        scores = np.asarray(S.sum(axis=1)).ravel()
        self._seen_mask(profile, scores)

        out = []
        for i in np.argsort(-scores)[:k]:
            if scores[i] <= 0:
                break
            sims = S.getrow(i).toarray().ravel()
            best = np.argsort(-sims)[:3]
            reasons = [{"movieId": used[j], "similarity": round(float(sims[j]), 3)}
                       for j in best if sims[j] > 0]
            out.append({"movieId": int(self.item_ids[i]), "score": float(scores[i]), "reasons": reasons})
        return out


# ---------------------------------------------------------------------------
# 3. User-to-user kNN
# ---------------------------------------------------------------------------

class UserKNN(BaseRecommender):
    """Find the training users whose liked movies overlap most with this profile, 
    then recommend what those neighbours liked."""

    name = "user_knn"

    def __init__(self, n_neighbors=50, fallback=True):
        self.n_neighbors = n_neighbors
        self.fallback = fallback

    def fit(self, train):
        self._prepare(train)
        self.U = self._liked_matrix(train).tocsr()      # binary, users x items
        self.Un = normalize(self.U)                     # rows have length 1
        self._pop = Popularity().fit(train)
        return self

    def _profile_vector(self, used):
        v = np.zeros(len(self.item_ids))
        v[[self.item_index[m] for m in used]] = 1.0
        return v

    def recommend(self, profile, k=10):
        return self._recommend(profile, k)

    def recommend_user(self, user_id, k=10):
        # a training user would be their own nearest neighbour, so leave them out
        return self._recommend(self.user_profiles.get(user_id, {}), k,
                               exclude_row=self.user_index.get(user_id))

    def _recommend(self, profile, k, exclude_row=None):
        used = self._liked_in_index(profile)
        if not used:
            return self._pop.recommend(profile, k) if self.fallback else []

        q = self._profile_vector(used)
        q_norm = q / np.linalg.norm(q)
        sims = self.Un @ q_norm                          # one similarity per training user
        if exclude_row is not None:
            sims[exclude_row] = 0.0

        n = min(self.n_neighbors, int((sims > 0).sum()))
        if n == 0:
            return self._pop.recommend(profile, k) if self.fallback else []
        nbrs = np.argpartition(-sims, n - 1)[:n]
        nbrs = nbrs[sims[nbrs] > 0]
        w = sims[nbrs]

        scores = np.asarray(self.U[nbrs].T @ w).ravel()  # similarity-weighted votes
        self._seen_mask(profile, scores)

        liked_idx = np.array([self.item_index[m] for m in used])
        Unb = self.U[nbrs]                               # neighbours x items
        out = []
        for i in np.argsort(-scores)[:k]:
            if scores[i] <= 0:
                break
            likes_i = np.asarray(Unb[:, i].todense()).ravel() > 0     # neighbours who liked item i
            shared = np.asarray(Unb[likes_i][:, liked_idx].T @ w[likes_i]).ravel()
            best = np.argsort(-shared)[:3]
            reasons = [{"movieId": used[j], "similarity": round(float(shared[j] / w[likes_i].sum()), 3)}
                       for j in best if shared[j] > 0]
            out.append({"movieId": int(self.item_ids[i]), "score": float(scores[i]),
                        "n_neighbors_liked": int(likes_i.sum()), "reasons": reasons})
        return out


# ---------------------------------------------------------------------------
# 4. Matrix factorization
# ---------------------------------------------------------------------------

class MatrixFactorization(BaseRecommender):

    name = "mf"

    def __init__(self, factors=32, reg=0.1, alpha=20.0, iterations=15, seed=42, fallback=True):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.fallback = fallback

    def fit(self, train):
        self._prepare(train)
        self._pop = Popularity().fit(train)
        R = self._liked_matrix(train).tocsr()    # users x items
        Rt = R.T.tocsr()                          # items x users
        n_users, n_items = R.shape
        rng = np.random.default_rng(self.seed)

        self.has_positive = np.asarray(Rt.sum(axis=1)).ravel() > 0   # items we can score
        self.X = rng.normal(0, 0.1, (n_users, self.factors))
        self.Y = rng.normal(0, 0.1, (n_items, self.factors))
        self.Y[~self.has_positive] = 0.0

        for _ in range(self.iterations):
            self.X = self._solve(R, self.Y)
            self.Y = self._solve(Rt, self.X)
            self.Y[~self.has_positive] = 0.0

        # pieces needed to fold in a new user later
        self._YtY = self.Y.T @ self.Y
        return self

    def _solve(self, P, F):
        FtF = F.T @ F
        I = np.eye(self.factors)
        out = np.zeros((P.shape[0], self.factors))
        for r in range(P.shape[0]):
            cols = P.indices[P.indptr[r]:P.indptr[r + 1]]
            if len(cols) == 0:
                continue
            Fp = F[cols]
            A = FtF + self.alpha * (Fp.T @ Fp) + self.reg * I
            b = (1.0 + self.alpha) * Fp.sum(axis=0)
            out[r] = np.linalg.solve(A, b)
        return out

    def recommend(self, profile, k=10):
        used = self._liked_in_index(profile)
        used = [m for m in used if self.has_positive[self.item_index[m]]]
        if not used:
            return self._pop.recommend(profile, k) if self.fallback else []

        idx = [self.item_index[m] for m in used]
        Yp = self.Y[idx]                                     # liked movie vectors
        A = self._YtY + self.alpha * (Yp.T @ Yp) + self.reg * np.eye(self.factors)
        Ainv_Yp = np.linalg.solve(A, Yp.T)                   # factors x n_liked
        contrib = (1.0 + self.alpha) * (self.Y @ Ainv_Yp)    # items x n_liked
        scores = contrib.sum(axis=1)
        scores[~self.has_positive] = -np.inf
        self._seen_mask(profile, scores)

        out = []
        for i in np.argsort(-scores)[:k]:
            if not np.isfinite(scores[i]):
                break
            c = contrib[i]
            best = np.argsort(-c)[:3]
            total = float(np.abs(c).sum()) or 1.0
            reasons = [{"movieId": used[j], "similarity": round(float(c[j] / total), 3)}
                       for j in best if c[j] > 0]
            out.append({"movieId": int(self.item_ids[i]), "score": float(scores[i]), "reasons": reasons})
        return out


MODELS = {
    "popularity": Popularity,
    "item_knn": ItemKNN,
    "user_knn": UserKNN,
    "mf": MatrixFactorization,
}