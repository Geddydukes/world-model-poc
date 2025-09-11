import os, sqlite3, numpy as np

class EpisodicMemory:
    def __init__(self, sqlite_path="memory/episodic.sqlite", embed_dir="memory/embeddings"):
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        os.makedirs(embed_dir, exist_ok=True)
        self.embed_dir = embed_dir
        self.conn = sqlite3.connect(sqlite_path)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS clips(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, path TEXT, modality TEXT, embed_path TEXT, dim INTEGER)""")
        self.conn.commit()

    def add_embedding(self, date, path, modality, vec):
        fname = f"{modality}_{date}_{abs(hash(path)) & 0xffffffff}.npy"
        fpath = os.path.join(self.embed_dir, fname)
        np.save(fpath, vec.astype(np.float32))
        cur = self.conn.cursor()
        cur.execute("INSERT INTO clips(date,path,modality,embed_path,dim) VALUES(?,?,?,?,?)",
                    (date, path, modality, fpath, int(vec.shape[-1])))
        self.conn.commit()

    def all_for_date(self, date, modality="image"):
        cur = self.conn.cursor()
        cur.execute("SELECT path, embed_path FROM clips WHERE date=? AND modality=?", (date, modality))
        rows = cur.fetchall()
        return [(p, np.load(ep)) for p, ep in rows]

    def nearest(self, query_vec, modality=None, topk=5):
        cur = self.conn.cursor()
        if modality: cur.execute("SELECT path, embed_path FROM clips WHERE modality=?", (modality,))
        else: cur.execute("SELECT path, embed_path FROM clips")
        rows = cur.fetchall()
        if not rows: return []
        mats, paths = [], []
        for p, ep in rows:
            mats.append(np.load(ep)); paths.append(p)
        M = np.vstack(mats); q = query_vec.astype(np.float32)
        Mn = M/(np.linalg.norm(M, axis=1, keepdims=True)+1e-8)
        qn = q/(np.linalg.norm(q)+1e-8)
        sims = Mn @ qn
        idx = np.argsort(-sims)[:topk]
        return [(paths[i], float(sims[i])) for i in idx]
