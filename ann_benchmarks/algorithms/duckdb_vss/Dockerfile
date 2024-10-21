FROM ann-benchmarks

RUN apt update -y

RUN apt install -y ninja-build cmake build-essential

RUN pip3 install numpy pyarrow

RUN git clone --recurse-submodules https://github.com/Maxxen/duckdb_vss.git

WORKDIR /home/app/duckdb_vss

COPY ann_benchmarks/algorithms/duckdb_vss/Makefile Makefile

RUN GEN=ninja make release

WORKDIR /home/app
