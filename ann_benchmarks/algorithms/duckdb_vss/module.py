import os


import duckdb
import numpy
import pyarrow as pa

from ..base.module import BaseANN


class DuckDBVSS(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._con = None

    def fit(self, X):

        width = X.shape[1]

        if self._metric == "angular":
            self._query = "SELECT id FROM items ORDER BY array_cosine_similarity(embedding, ?::FLOAT[%d]) LIMIT ?" % width
        elif self._metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY array_distance(embedding, ?::FLOAT[%d]) LIMIT ?" % width
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

        # Delete the database file if it exists
        if os.path.exists("temp.db"):
            os.remove("temp.db")

        # Connect to the database
        con = duckdb.connect("temp.db")
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        con.execute("DROP TABLE IF EXISTS items")
        con.execute("CREATE TABLE items (id int, embedding FLOAT[%d])" % width)

        print("copying data...")
        row_id_array = pa.array(range(X.shape[0]))
        arr = numpy.copy(X)
        arr.shape = -1
        embedding_array = pa.FixedSizeListArray.from_arrays(arr, width)
        embedding_table = pa.Table.from_arrays([row_id_array, embedding_array], ['id', 'embedding'])
        
        con.execute(f"INSERT INTO items SELECT id, embedding FROM embedding_table ORDER BY id")

        print("creating index...")
        if self._metric == "angular":
            con.execute("CREATE INDEX my_idx ON items USING HNSW (embedding) WITH (metric='cosine', m = %d, ef_construction = %d)" % (self._m, self._ef_construction))
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
        mem = self._con.execute("SELECT approx_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'my_idx'").fetchone()
        return mem[0] / 1024

    def __str__(self):
        return f"DuckDBVSS(m={self._m}, ef_construction={self._ef_construction})"
