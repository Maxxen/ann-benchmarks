import numpy
import duckdb

from ..base.module import BaseANN


class DuckDBVSS(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._con = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY array_cosine_similarity(embedding, %s) LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY array_distance(embedding, %s) LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        #subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        #conn = psycopg.connect(user="ann", password="ann", dbname="ann", autocommit=True)
        #pgvector.psycopg.register_vector(conn)
        con = duckdb.connect()
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        con.execute("DROP TABLE IF EXISTS items")
        con.execute("CREATE TABLE items (id int, embedding FLOAT[%d])" % X.shape[1])

        # DuckDB only supports float vectors
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        print("copying data...")
        for i, embedding in enumerate(X):
            con.execute("INSERT INTO items VALUES (?, ?)", (i, embedding))
            
        print("creating index...")
        if self._metric == "angular":
            con.execute(
                "CREATE INDEX my_idx ON items USING HNSW (embedding) WITH (metric='cosine', m = %d, ef_construction = %d)" % (self._m, self._ef_construction)
            )
        elif self._metric == "euclidean":
            con.execute("CREATE INDEX my_idx ON items USING HNSW (embedding) WITH (metric='l2sq', m = %d, ef_construction = %d)" % (self._m, self._ef_construction))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._con = con

    def set_query_arguments(self, ef_search):
        pass
        #self._ef_search = ef_search
        #self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        res = self._con.execute(self._query, (v, n)).fetchall()
        return [id for id, in res]
    
    def get_memory_usage(self):
        if self._con is None:
            return 0
        mem = self._con.execute("SELECT approx_memory_usage FROM pragma_hnsw_index_info() WHERE name = 'my_idx'").fetchone()
        return mem[0] / 1024

    def __str__(self):
        return f"DuckDBVSS(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
